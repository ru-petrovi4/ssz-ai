using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models
{
    public static class MiniColumnsSyncronization
    {
        public static bool TrainSyncronization(Cortex.MiniColumn miniColumn, float[] shortHashConverted)
        {
            if (miniColumn.Temp_ShortHashConversionMatrix_TrainingCount < 200)
            {
                float oneProbability = 1.0f / (float)miniColumn.Constants.ShortHashBitsCount;
                float zeroProbability = 1.0f / ((float)miniColumn.Constants.ShortHashLength - (float)miniColumn.Constants.ShortHashBitsCount);
                foreach (int j in Enumerable.Range(0, shortHashConverted.Length))
                    foreach (int i in Enumerable.Range(0, miniColumn.Temp_ShortHash.Length))
                    {
                        //miniColumn.Temp_ShortHashConversionMatrix![i, j] -= avarageProbability;
                        if (miniColumn.Temp_ShortHash[i] == 1.0f)
                        {
                            if (shortHashConverted[j] == 1.0f)
                                miniColumn.Temp_ShortHashConversionMatrix![i, j] += oneProbability;
                            //else
                                //miniColumn.Temp_ShortHashConversionMatrix![i, j] -= 2 * zeroProbability;
                        }
                        else
                        {
                            if (shortHashConverted[j] == 0.0f)
                                miniColumn.Temp_ShortHashConversionMatrix![i, j] += zeroProbability;
                            //else
                                //miniColumn.Temp_ShortHashConversionMatrix![i, j] -= 2 * oneProbability;
                        }
                    }

                miniColumn.Temp_ShortHashConversionMatrix_TrainingCount += 1;

                if (miniColumn.Temp_ShortHashConversionMatrix_TrainingCount == 200)
                {
                    foreach (int i in Enumerable.Range(0, miniColumn.Temp_ShortHash.Length))
                    {
                        int jMax = 0;
                        float max = miniColumn.Temp_ShortHashConversionMatrix![i, 0];

                        foreach (int j in Enumerable.Range(0, miniColumn.Temp_ShortHashConversionMatrix!.Dimensions[1]))
                        {
                            float v = miniColumn.Temp_ShortHashConversionMatrix![i, j];
                            if (v > max)
                            {
                                max = v;
                                jMax = j;
                            }
                        }

                        miniColumn.ShortHashConversion[i] = jMax;
                        foreach (int i2 in Enumerable.Range(0, miniColumn.Temp_ShortHash.Length))
                        {
                            miniColumn.Temp_ShortHashConversionMatrix![i2, jMax] = 0.0f;
                        }
                    }
                    miniColumn.Temp_SyncQualitySum = 0.0f;
                    miniColumn.Temp_SyncQualitySumCount = 0;
                }
            }            
            else
            {
                miniColumn.CalculateShortHashConverted(miniColumn.Temp_ShortHash, miniColumn.Temp_ShortHashConverted);
                miniColumn.Temp_SyncQualitySum += TensorPrimitives.CosineSimilarity(miniColumn.Temp_ShortHashConverted, shortHashConverted);
                miniColumn.Temp_SyncQualitySumCount += 1;
                miniColumn.Temp_SyncQuality = miniColumn.Temp_SyncQualitySum / miniColumn.Temp_SyncQualitySumCount;
            }

            return false;
        }
    }
}
