import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import pandas as pd
import torch

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}

results_path = 'evaluated_results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0'

def main():
    run(results_path)


def run(results_path, gpu=[0], seed=0,classname='bottle', demo=False):

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )


    result_collect = []

    if demo:
        classname = classname + '_captured'
    dataloader_iter, n_dataloaders = dataset(name='mvtec',data_path='./dataset/',subdatasets=[classname])
    dataloader_iter = dataloader_iter(seed)

    patch_core_paths=['./results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0/models/mvtec_bottle']
    patchcore_iter, n_patchcores = patch_core_loader(patch_core_paths)
    patchcore_iter = patchcore_iter(device)

    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )
    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, labels_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                # aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)


            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            if demo:
                return scores

            if not demo:
                LOGGER.info("Computing evaluation metrics.")
                # Compute Image-level AUROC scores for all images.
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["auroc"]
                accuracy = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["accuracy"]
                optimal_threshold = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["optimal_threshold"]

                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "instance_auroc": auroc,
                        "accuracy": accuracy,
                        "optimal_threshold":optimal_threshold,
                        # "full_pixel_auroc": full_pixel_auroc,
                        # "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    }
                )

                for key, item in result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:3.3f}".format(key, item))

                del PatchCore_list
                gc.collect()

        LOGGER.info("\n\n-----\n")

        # df = pd.DataFrame(labels_gt,columns=['GT'])
        # df['PRED'] = predictions
        # df.to_excel('evaluated_results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0/predictions.xlsx')

    if not demo:
        result_metric_names = list(result_collect[-1].keys())[1:]
        result_dataset_names = [results["dataset_name"] for results in result_collect]
        result_scores = [list(results.values())[1:] for results in result_collect]
        patchcore.utils.compute_and_store_final_results(
            results_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_dataset_names,
        )



######### patchcores loading ###############
def patch_core_loader(patch_core_paths, faiss_on_gpu=True, faiss_num_workers=8):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return (get_patchcore_iter, len(patch_core_paths))

############## dataset ############
def dataset(
    name, data_path, subdatasets, batch_size=1, resize=366, imagesize=320, num_workers=8, augment=True
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return (get_dataloaders_iter, len(subdatasets))


if __name__ == '__main__':
    main()