using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models
{
    public static class MiniColumnsSyncronization
    {
        public static bool TrainSyncronization(Cortex.MiniColumn miniColumn, float[] shortHashConverted)
        {
            if (miniColumn.Temp_ShortHashConversionMatrix_TrainingCount < 700)
            {
                float oneProbability = 1.0f / (float)miniColumn.Constants.ShortHashBitsCount;
                float zeroProbability = 1.0f / ((float)miniColumn.Constants.ShortHashLength - (float)miniColumn.Constants.ShortHashBitsCount);
                foreach (int j in Enumerable.Range(0, shortHashConverted.Length))
                    foreach (int i in Enumerable.Range(0, miniColumn.Autoencoder!.Temp_ShortHash.Length))
                    {
                        //miniColumn.Temp_ShortHashConversionMatrix![i, j] -= avarageProbability;
                        if (miniColumn.Autoencoder!.Temp_ShortHash[i] == 1.0f)
                        {
                            if (shortHashConverted[j] == 1.0f)
                                miniColumn.Temp_ShortHashConversionMatrix![i, j] += oneProbability;
                            //else
                            //    miniColumn.Temp_ShortHashConversionMatrix![i, j] -= 2 * zeroProbability;
                        }
                        else
                        {
                            if (shortHashConverted[j] == 0.0f)
                                miniColumn.Temp_ShortHashConversionMatrix![i, j] += zeroProbability;
                            //else
                            //    miniColumn.Temp_ShortHashConversionMatrix![i, j] -= 2 * oneProbability;
                        }
                    }

                miniColumn.Temp_ShortHashConversionMatrix_TrainingCount += 1;

                if (miniColumn.Temp_ShortHashConversionMatrix_TrainingCount == 200)
                {
                    //var shortHashConversionMatrix = miniColumn.Temp_ShortHashConversionMatrix!.Clone();
                    //foreach (int i in Enumerable.Range(0, miniColumn.Autoencoder!.Temp_ShortHash.Length))
                    //{
                    //    int jMax = 0;
                    //    float max = shortHashConversionMatrix[i, 0];

                    //    foreach (int j in Enumerable.Range(0, shortHashConversionMatrix.Dimensions[1]))
                    //    {
                    //        float v = shortHashConversionMatrix[i, j];
                    //        if (v > max)
                    //        {
                    //            max = v;
                    //            jMax = j;
                    //        }
                    //    }

                    //    miniColumn.ShortHashConversion[i] = jMax;
                    //    foreach (int i2 in Enumerable.Range(0, miniColumn.Autoencoder!.Temp_ShortHash.Length))
                    //    {
                    //        shortHashConversionMatrix[i2, jMax] = float.MinValue;
                    //    }
                    //}
                    //miniColumn.Temp_ShortHashConverted_SyncQualitySum = 0.0f;
                    //miniColumn.Temp_ShortHashConverted_SyncQualitySumCount = 0;
                }
            }            
            else
            {
                miniColumn.GetShortHashConverted(miniColumn.Autoencoder!.Temp_ShortHash, miniColumn.Temp_ShortHashConverted);
                miniColumn.Temp_ShortHashConverted_SyncQualitySum += TensorPrimitives.CosineSimilarity(miniColumn.Temp_ShortHashConverted, shortHashConverted);
                miniColumn.Temp_ShortHashConverted_SyncQualitySumCount += 1;
                miniColumn.Temp_ShortHashConverted_SyncQuality = miniColumn.Temp_ShortHashConverted_SyncQualitySum / miniColumn.Temp_ShortHashConverted_SyncQualitySumCount;
            }

            return false;
        }
    }
}
