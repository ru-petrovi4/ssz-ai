using System.Linq;

namespace Ssz.AI.Models
{
    public static class MiniColumnsSyncronization
    {
        public static bool TrainSyncronization(Cortex.MiniColumn miniColumn, float[] shortHash, float[] shortHashConverted)
        {
            float oneProbability = 1.0f / (float)miniColumn.Constants.ShortHashBitsCount;
            float zeroProbability = 1.0f / ((float)miniColumn.Constants.ShortHashLength - (float)miniColumn.Constants.ShortHashBitsCount);            
            foreach (int j in Enumerable.Range(0, shortHashConverted.Length))
                foreach (int i in Enumerable.Range(0, shortHash.Length))
                {
                    //miniColumn.Temp_ShortHashConversionMatrix![i, j] -= avarageProbability;
                    if (shortHash[i] == 1.0f)
                    {
                        if (shortHashConverted[j] == 1.0f)
                            miniColumn.Temp_ShortHashConversionMatrix![i, j] += oneProbability;                        
                    }
                    else
                    {
                        if (shortHashConverted[j] == 0.0f)
                            miniColumn.Temp_ShortHashConversionMatrix![i, j] += zeroProbability;
                    }
                }
            
            return false;
        }
    }
}
