import torch

def kmeans(data, K, max_iters=100):
    # Step 1: Initialize centers (K-means++ or random)
    centers = data[torch.randperm(data.size(0))[:K]]
    
    for _ in range(max_iters):
        # Step 2: Assign data points to nearest centers
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)
        
        # Step 3: Update centers
        new_centers = torch.stack([data[labels == k].mean(dim=0) for k in range(K)])
        
        # Check for convergence
        if torch.all(new_centers == centers):
            break
        
        centers = new_centers
    
    return centers
    # 计算每个聚类中心的密集度并按密集度从大到小排序
    centroid_densities = [torch.mean(torch.sum((data[labels == i] - centers[i])**2, dim=1)) for i in range(K)]
    centroid_densities = torch.tensor(centroid_densities).cuda()
    sorted_indices = torch.argsort(centroid_densities, descending=True)

    # 按排序顺序输出聚类中心和对应的密集度
    sorted_centroids = centers[sorted_indices]
    sorted_densities = torch.tensor(centroid_densities)[sorted_indices]

    # for i, (centroid, density) in enumerate(zip(sorted_centroids, sorted_densities)):
    #     print(f"聚类中心 {i+1}: {centroid.cpu().numpy()}, 密集度: {density:.2f}")
    
    return sorted_centroids