using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model
    {
        public void Calculate_Clusterization_Algorithm_Random(ILoggersSet loggersSet)
        {
            var totalStopwatch = Stopwatch.StartNew();

            for (int wordIndex = 0; wordIndex < Words_RU.Count; wordIndex += 1)
            {
                Words_RU[wordIndex].Temp_Flag = false;
            }

            Random r = new();

            var primaryWords_Random = new Word[PrimaryWordsCount];
            for (int index = 0; index < primaryWords_Random.Length; index += 1)
            {
                for (; ; )
                {
                    var word = Words_RU[r.Next(Words_RU.Count)];
                    if (word.Temp_Flag)
                        continue;

                    primaryWords_Random[index] = word;
                    word.Temp_Flag = true;
                    break;
                }
            }
            Clusterization_Algorithm_Random.PrimaryWords = primaryWords_Random;

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("CalculatePrimaryWords_Random totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }        
    }    
}

//public float[] DotProducts = null!;