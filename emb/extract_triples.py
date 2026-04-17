import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from tqdm import tqdm


def extract_triples_from_webqsp(output_file='webqsp_triples.txt'):
    """
    从webqsp数据集中提取所有三元组，去重后保存到txt文件

    Parameters:
    -----------
    output_file : str
        输出文件路径
    """
    # 加载webqsp数据集的所有分割
    print("正在加载webqsp数据集...")
    train_set = load_dataset('ml1996/webqsp', split='train')
    val_set = load_dataset('ml1996/webqsp', split='validation')
    test_set = load_dataset('ml1996/webqsp', split='test')

    # 使用集合来去重三元组
    triples_set = set()

    # 处理训练集
    print("正在处理训练集...")
    for sample in tqdm(train_set):
        graph = sample['graph']
        for triple in graph:
            head, relation, tail = triple
            # 将三元组转换为元组以便加入集合去重
            triples_set.add((head, relation, tail))

    # 处理验证集
    print("正在处理验证集...")
    for sample in tqdm(val_set):
        graph = sample['graph']
        for triple in graph:
            head, relation, tail = triple
            triples_set.add((head, relation, tail))

    # 处理测试集
    print("正在处理测试集...")
    for sample in tqdm(test_set):
        graph = sample['graph']
        for triple in graph:
            head, relation, tail = triple
            triples_set.add((head, relation, tail))

    # 写入文件
    print(f"正在写入文件 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for head, relation, tail in sorted(triples_set):
            f.write(f"{head}\t{relation}\t{tail}\n")

    print(f"完成！共提取 {len(triples_set)} 条不重复的三元组")
    print(f"已保存到: {output_file}")


if __name__ == '__main__':
    # 可以指定输出文件路径
    extract_triples_from_webqsp('webqsp_triples.txt')