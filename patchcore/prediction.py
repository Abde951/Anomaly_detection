import pandas as pd
from evaluation import run


def main():
    ## a test for bottle dataset
    compute_optimal_score('bottle')
    pred = prediction('bottle')
    print(pred)

def compute_optimal_score(classname):
    results_path = 'evaluated_results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0'
    ## computing optimal threshold using what we have as dataset
    run(results_path,classname=classname)


def prediction(classname):
    results_path = 'evaluated_results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0'
    ## computing scores of new captured image
    score = run(results_path,demo=True,classname=classname)
    
    df = pd.read_csv(results_path + '/results.csv')
    optimal_threshold = df['optimal_threshold'][0]
    print(optimal_threshold)

    predicted = (score >= optimal_threshold).astype(int)
    
    return predicted



if __name__ == '__main__':
    main()