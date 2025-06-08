using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using static Ssz.AI.Models.Cortex_Simplified;

namespace Ssz.AI.Models
{
    public static class Clusterization
    {
        public static List<List<Memory>> GetMemoryClusters(List<Memory?> memories, IConstants constants)
        {
            List<List<Memory>> memoryClusters = memories.Where(m => m is not null).Select(m => new List<Memory> { m! }).ToList();

            while (memoryClusters.Count > 1)
            {
                float minDistance = float.MaxValue;
                int clusterA = -1, clusterB = -1;

                for (int i = 0; i < memoryClusters.Count; i++)
                {
                    for (int j = i + 1; j < memoryClusters.Count; j++)
                    {
                        float distance = AverageLinkage(memoryClusters[i], memoryClusters[j]);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            clusterA = i;
                            clusterB = j;
                        }
                    }
                }

                if (minDistance > constants.MemoryClustersThreshold) 
                    break; // Остановить объединение, если превышен порог

                // Объединение двух ближайших кластеров
                memoryClusters[clusterA].AddRange(memoryClusters[clusterB]);
                memoryClusters.RemoveAt(clusterB);
            }

            return memoryClusters;
        }

        private static float AverageLinkage(List<Memory> cluster1, List<Memory> cluster2)
        {
            float totalDistance = 0;
            int comparisons = 0;

            foreach (var v1 in cluster1)
            {
                foreach (var v2 in cluster2)
                {
                    totalDistance += 1.0f - TensorPrimitives.CosineSimilarity(v1.Hash, v2.Hash);
                    comparisons += 1;
                }
            }
            return totalDistance / comparisons;
        }
    }
}
