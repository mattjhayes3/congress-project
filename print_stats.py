import glob
import json
import pandas as pd

if __name__ == "__main__":
    stats_per_model = dict()
    all_stats = dict()
    for file in glob.glob('models/*/*_stats_*.json'):
        with open(file, 'r') as f:
            stat = json.load(f)
            stat['source'] = file
            stat['model'] = stat['model'].replace('models.', '')
            if stat['split'] == 'test' and 'models/RF_CV_PER_' not in stat['source']:
                if stat['model'] not in stats_per_model:
                    stats_per_model[stat['model']] = dict()
                assert stat['dataset'] not in stats_per_model[stat['model']], f"{stat['dataset']} already in {stats_per_model[stat['model']]}, path={file}"
                stats_per_model[stat['model']][stat['dataset']] = stat
                print(f"found model={stat['model']}, dataset={stat['dataset']}, path={file}")
        if len(all_stats) == 0:
            for k,v in stat.items():
                all_stats[k] = [v]
        else:
            if stat.keys() != all_stats.keys():
                print(f"skipping {file}, {stat.keys()} not the same as {all_stats.keys()}")
                continue
            for k,v in stat.items():
                all_stats[k].append(v)

    df = pd.DataFrame.from_dict(all_stats)
    print(df)
    # log_stats = df[(df['model'] == 'models.logistic') & (df['split'] == 'test')]
    test_stats = df[(df['split'] == 'test') & (df['model'] != 'dem_baseline') & (df['model'] != 'rep_baseline')].sort_values(by=['dataset', 'model'])
    print("test stats:")
    print(test_stats)
    # test_stats = df[df['split'] == 'test']
    # log_test_stats = log_stats * test_stats
    with pd.option_context('display.max_rows', None):
        print(test_stats[['dataset', 'model', 'accuracy', 'recall', 'negative_accuracy', 'source']])

    dsets = list(set(all_stats['dataset']))
    dsets.sort()
    print("dsets=", dsets)
    models = list(set(all_stats['model']))
    models.sort()
    bayram_models = ['rf_bayram', 'svm_bayram', 'logistic_bayram', 'nn20nd_bayram', 'nn20d_bayram', 'nn1000nd_bayram', 'nn1000d_bayram']
    acc_model_by_dataset = {
        'model': bayram_models.copy(),
        'h097': [0.605, 0.623, 0.615, 0.573, 0.619, 0.566, 0.611],
        'h100': [0.620, 0.639, 0.640, 0.597, 0.647, 0.585, 0.635],
        'h103': [0.672, 0.711, 0.701, 0.671, 0.700, 0.669, 0.698],
        'h106': [0.662, 0.685, 0.675, 0.638, 0.678, 0.635, 0.674],
        'h109': [0.739, 0.754, 0.755, 0.704, 0.747, 0.689, 0.744],
        'h112': [0.759, 0.783, 0.774, 0.739, 0.780, 0.731, 0.777],
        'h114': [0.735, 0.779, 0.772, 0.738, 0.774, 0.724, 0.768],
    }
    pos_acc_model_by_dataset = {
        'model': bayram_models.copy(),
        'h097': [0.619, 0.639, 0.596, 0.606, 0.607, 0.600, 0.641],
        'h100': [0.613, 0.658, 0.600, 0.561, 0.677, 0.613, 0.621],
        'h103': [0.671, 0.722, 0.710, 0.699, 0.717, 0.663, 0.696],
        'h106': [0.699, 0.696, 0.667, 0.657, 0.678, 0.642, 0.672],
        'h109': [0.701, 0.730, 0.751, 0.708, 0.699, 0.697, 0.749],
        'h112': [0.751, 0.779, 0.767, 0.735, 0.742, 0.738, 0.777],
        'h114': [0.705, 0.778, 0.771, 0.743, 0.724, 0.734, 0.766],
    }
    neg_acc_model_by_dataset = {'model': bayram_models.copy(), 'h097':[], 'h100':[], 'h103':[],'h106':[],'h109':[],'h112':[],'h114':[]}
    for i in range(len(bayram_models)):
        for ds in dsets:
            neg_acc = acc_model_by_dataset[ds][i] * 2 - pos_acc_model_by_dataset[ds][i]
            neg_acc_model_by_dataset[ds].append(neg_acc)
    # print(f"len acc models: {len(acc_model_by_dataset['model'])}")
    # print(f"len adding models: {len(models)}")
    for model in models:
        if not 'baseline' in model and model not in ['rf', 'sk_logistic']:
            # print(f"len acc models: {len(acc_model_by_dataset['model'])}")
            acc_model_by_dataset['model'].append(model)
            pos_acc_model_by_dataset['model'].append(model)
            neg_acc_model_by_dataset['model'].append(model)
            for ds in dsets:
                # print("gathering", model, "ds" ,ds)
                assert ds in acc_model_by_dataset
                assert model in stats_per_model
                assert ds in stats_per_model[model], stats_per_model[model]
                acc_model_by_dataset[ds].append(stats_per_model[model][ds]['accuracy'])
                pos_acc_model_by_dataset[ds].append(stats_per_model[model][ds]['recall'])
                neg_acc_model_by_dataset[ds].append(stats_per_model[model][ds]['negative_accuracy'])

    acc_model_by_dataset['average'] = []
    pos_acc_model_by_dataset['average'] = []
    neg_acc_model_by_dataset['average'] = []
    for i in range(len(acc_model_by_dataset['model'])):
        acc = 0
        pos = 0
        neg = 0
        for ds in dsets:
            acc += acc_model_by_dataset[ds][i]
            pos += pos_acc_model_by_dataset[ds][i]
            neg += neg_acc_model_by_dataset[ds][i]
        acc_model_by_dataset['average'].append(acc/len(dsets))
        pos_acc_model_by_dataset['average'].append(pos/len(dsets))
        neg_acc_model_by_dataset['average'].append(neg/len(dsets))
    # print(f"acc_model_by_dataset.keys()={acc_model_by_dataset.keys()}")
    # for k,v in acc_model_by_dataset.items():
    #     print(f"len {k} = {len(v)}")
    # print(acc_model_by_dataset)
    # for stat in stats_list:
    #     # print(stat['source'])
    #         stats_per_model
    #         if not stat['dataset'] in acc_model_by_dataset:
    #             acc_model_by_dataset[stat['dataset']] = []
    #             pos_acc_model_by_dataset[stat['dataset']] = []
    #             neg_acc_model_by_dataset[stat['dataset']] = []
    #         acc_model_by_dataset['model'].append(stat['model'])
    #         acc_model_by_dataset[stat['dataset']].append(stat['accuracy'])
    #         pos_acc_model_by_dataset['model'].append(stat['model'])
    #         pos_acc_model_by_dataset[stat['dataset']].append(stat['recall'])
    #         neg_acc_model_by_dataset['model'].append(stat['model'])
    #         neg_acc_model_by_dataset[stat['dataset']].append(stat['negative_accuracy'])

    # acc_model_by_dataset = {ds: [] for ds in dsets}
    # acc_model_by_dataset['model'] = []
    # pos_acc_model_by_dataset = {ds: [] for ds in dsets}
    # pos_acc_model_by_dataset['model'] = []
    # neg_acc_model_by_dataset = {ds: [] for ds in dsets}
    # neg_acc_model_by_dataset['model'] = []
    dsets.append('average')

    acc_model_by_dataset = pd.DataFrame.from_dict(acc_model_by_dataset).sort_values(by=['model'], ascending=False)
    pos_acc_model_by_dataset = pd.DataFrame.from_dict(pos_acc_model_by_dataset).sort_values(by=['model'], ascending=False)
    neg_acc_model_by_dataset = pd.DataFrame.from_dict(neg_acc_model_by_dataset).sort_values(by=['model'], ascending=False)
    acc_maxes = acc_model_by_dataset[dsets].max()
    pos_maxes = pos_acc_model_by_dataset[dsets].max()
    neg_maxes = neg_acc_model_by_dataset[dsets].max()
    correspondence = {m:f"{m}_bayram" for m in models if "{m}_bayram" in models}
    tol = 0.02
    print("maxes", acc_maxes)
    def get_fmt(maxes, ds, x):
        return "%0.3f" % x
        
        # print(f"max for {ds} is {maxes[ds]}")
        # if x == maxes[ds]:
        #     fmt = "\\text_bf{%s}" % fmt
        # return fmt

    print(f"dsets={dsets}")
    acc_formatters = {ds: lambda x: get_fmt(acc_maxes, ds, x) for ds in dsets}
    acc_formatters['model'] = lambda x: x.upper().replace('_', ' ')
    print(acc_formatters)
    print('accuracy')
    # print(acc_model_by_dataset)
    s = acc_model_by_dataset.style
    # s.highlight_max(props='font-weight:bold;')
    # s.format({
    #     ("Numeric", "Floats"): '{:.3f}',
    #     ("Non-Numeric", "Strings"): str.upper,
    # })

    # print(acc_model_by_dataset.style.to_latex(formatters=acc_formatters))
    print(acc_model_by_dataset.to_latex(index=False, formatters = acc_formatters, caption="(Balanced) accuracy for preliminary bag-of-words models.  Maximal values for each dataset are bolded.  For models that have a corresponding baseline in Bayram et al, increased values are shown in green, decreased in red."))
    print('positive accuracy')
    # print(pos_acc_model_by_dataset)
    print(pos_acc_model_by_dataset.to_latex(index=False, formatters = acc_formatters, caption="Democrat accuracy for preliminary bag-of-words models.  Maximal values for each dataset are bolded.  For models that have a corresponding baseline in Bayram et al, increased values are shown in green, decreased in red."))
    print()
    print('negative accuracy')
    # print(neg_acc_model_by_dataset)
    print(neg_acc_model_by_dataset.to_latex(index=False, formatters = acc_formatters, caption="Republican accuracy for preliminary bag-of-words models.  Maximal values for each dataset are bolded.  For models that have a corresponding baseline in Bayram et al, increased values are shown in green, decreased in red."))
    print()
