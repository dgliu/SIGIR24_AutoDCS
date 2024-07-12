import itertools

from evaluator.metrics import metrics_dict
from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout, eval_score_matrix_foldout_ndcg
import multiprocessing
import heapq
import numpy as np
cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
data_generator2 = Data(path=args.data_path + args.dataset2, batch_size=args.batch_size)
data_generator3 = Data(path=args.data_path + args.dataset3, batch_size=args.batch_size)
data_generator4 = Data(path=args.data_path + args.dataset4, batch_size=args.batch_size)

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size


def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    # return np.array([idx for ele, idx in ele_idx], dtype=np.intc)
    return np.array(ele_idx)[:,-1].astype(np.intc)


def test_back(sess, model, users_to_test, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of items
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = { 'hit': np.zeros(len(model.Ks)), 'ndcg': np.zeros(len(model.Ks)), 'precision': np.zeros(len(model.Ks))}
    hr = {str(top_show[0]): np.zeros(len(top_show)), str(top_show[1]): np.zeros(len(top_show)), str(top_show[2]): np.zeros(len(top_show)), str(top_show[3]): np.zeros(len(top_show))}
    
    u_batch_size = BATCH_SIZE

    n_items_test = 0
    hit_sum = 0
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    count = 0
    all_result = []
    ndcg_result = []
    hr_result = []
    item_batch = range(ITEM_NUM)
    for u_batch_id in range(n_user_batchs): 
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch,
                                                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size))})
        rate_batch = np.array(rate_batch)
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                
                test_items.append(data_generator.test_set[user])
                n_items_test += len(data_generator.test_set[user])
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.    
            for idx, user in enumerate(user_batch):
                    train_items_off = data_generator.train_items[user]
                    rate_batch[idx][train_items_off] = -np.inf
        elif train_set_flag == 2:
            for user in user_batch:

                test_items.append(data_generator.valid_set[user])
                n_items_test += len(data_generator.valid_set[user])
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            # for idx, user in enumerate(user_batch):
            #    train_items_off = data_generator.train_items[user]
            #    rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                test_items.append(data_generator.train_items[user])
                n_items_test += len(data_generator.train_items[user])

        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)

        hr_result.append(batch_result)
        batch_ndcg = eval_score_matrix_foldout_ndcg(rate_batch, test_items, max_top)

        ndcg_result.append(batch_ndcg)

    final_hr_result = np.concatenate(hr_result, axis=0)
    final_hr_result = np.cumsum(final_hr_result, axis=0)[-1]

    for i in range(len(top_show)):
        hit_sum = final_hr_result[top_show[i] - 1]

        hr[str(top_show[i])] = hit_sum / n_items_test
    HR = []
    for i in range(len(hr)):
        HR.append(hr[str(top_show[i])])

    all_result = np.concatenate(ndcg_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean
    final_result = np.reshape(final_result, newshape=[1, max_top])
    final_result = final_result[:, top_show-1]
    final_result = np.reshape(final_result, newshape=[1, len(top_show)])
    final_result = final_result.flatten()
    print(final_result)

    result['hit'] += HR
    result['ndcg'] += final_result
    return result


def test_crgcn(sess, model, users_to_test, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of items
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = { 'hit': np.zeros(len(model.Ks)), 'ndcg': np.zeros(len(model.Ks)), 'precision': np.zeros(len(model.Ks))}
    hr = {str(top_show[0]): np.zeros(len(top_show)), str(top_show[1]): np.zeros(len(top_show)), str(top_show[2]): np.zeros(len(top_show)), str(top_show[3]): np.zeros(len(top_show))}
    
    u_batch_size = BATCH_SIZE

    n_items_test = []
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    topk_list = []
    item_batch = range(ITEM_NUM)
    for u_batch_id in range(n_user_batchs): 
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                        model.pos_items: item_batch,
                                                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size))})
        rate_batch = np.array(rate_batch)
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                
                test_items.append(data_generator.test_set[user])
                n_items_test.append(len(data_generator.test_set[user]))
            for idx, user in enumerate(user_batch):
                    train_items_off = data_generator.train_items[user]
                    rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                test_items.append(data_generator.train_items[user])
                n_items_test.append(len(data_generator.train_items[user]))
       
        for idx, user in enumerate(user_batch):
            topk_idx = argmax_top_k(rate_batch[idx], top_k=max_top)
            gt_items = test_items[idx]
            mask = np.isin(topk_idx, gt_items)
            topk_list.append(mask)

    test_gt_length = np.array(n_items_test)
    topk_list = np.array(topk_list)
    metric_dict = calculate_result(topk_list, test_gt_length, top_show)

    result['hit'] += np.array(metric_dict['hit'])
    result['ndcg'] += np.array(metric_dict['ndcg'])
    return result


def calculate_result(topk_list, gt_len, top_show):
    result_list = []
    metrics = ['hit', 'ndcg']
    for metric in metrics:
        metric_fuc = metrics_dict[metric.lower()]
        result = metric_fuc(topk_list, gt_len)
        result_list.append(result)
    result_list = np.stack(result_list, axis=0).mean(axis=1)

    metric_dict = {}
    final_result = []
    for topk in top_show:
        for metric, value in zip(metrics, result_list):
            key = metric
            metric_dict.setdefault(key,[])
            metric_dict[key].append(np.round(value[topk - 1], 4))

    return metric_dict


if args.shencha == 0:
    test = test_back
else:
    # print('check mode')
    test = test_crgcn