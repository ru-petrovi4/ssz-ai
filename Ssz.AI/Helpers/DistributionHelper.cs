using System;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Helpers
{
    public static class DistributionHelper
    {
        public static UInt64[] GetAccumulativeDistribution(UInt64[] distribution)
        {
            UInt64[] result = new UInt64[distribution.Length];
            UInt64 value = 0;
            foreach (int i in Enumerable.Range(0, distribution.Length))
            {
                value += distribution[i];
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        ///     Returns first valid index, where accumulativeDistribution[index] >= value or last element
        ///     <para>Precondition: accumulativeDistribution.Length > 0</para>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="accumulativeDistribution"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(UInt64 value, UInt64[] accumulativeDistribution)
        {
            int left = 0;
            int right = accumulativeDistribution.Length;

            while (left < right)
            {
                int mid = (left + right) >> 1; // Быстрое деление на 2 через сдвиг
                if (accumulativeDistribution[mid] >= value)
                    right = mid;
                else
                    left = mid + 1;
            }            
            if (right >= accumulativeDistribution.Length)
                right = accumulativeDistribution.Length - 1;
            return right;
        }

        /// <summary>
        ///     Precondition: accumulativeDistribution.Length > 0
        /// </summary>
        /// <param name="accumulativeDistribution"></param>
        /// <returns></returns>
        public static int GetRandom(Random random, UInt64[] accumulativeDistribution)
        {   
            return GetIndex((ulong)(random.NextDouble() * accumulativeDistribution[^1]), accumulativeDistribution);
        }
        
        public static float[] GetAccumulativeDistribution(float[] distribution)
        {
            float[] result = new float[distribution.Length];
            float value = 0;
            foreach (int i in Enumerable.Range(0, distribution.Length))
            {
                value += distribution[i];
                result[i] = value;
            }
            return result;
        }        

        /// <summary>
        ///     Returns first valid index, where accumulativeDistribution[index] >= value or last element
        ///     <para>Precondition: accumulativeDistribution.Length > 0</para>
        /// </summary>
        /// <param name="value"></param>
        /// <param name="accumulativeDistribution"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(float value, float[] accumulativeDistribution)
        {
            int left = 0;
            int right = accumulativeDistribution.Length;

            while (left < right)
            {
                int mid = (left + right) >> 1; // Быстрое деление на 2 через сдвиг
                if (accumulativeDistribution[mid] >= value)
                    right = mid;
                else
                    left = mid + 1;
            }
            if (right >= accumulativeDistribution.Length)
                right = accumulativeDistribution.Length - 1;
            return right;
        }

        /// <summary>
        ///     Precondition: accumulativeDistribution.Length > 0
        /// </summary>
        /// <param name="accumulativeDistribution"></param>
        /// <returns></returns>        
        public static int GetRandom(Random random, float[] accumulativeDistribution)
        {           
            return GetIndex(random.NextSingle() * accumulativeDistribution[^1], accumulativeDistribution); // ^1 Последний элемент массива
        }

        /// <summary>
        ///     Returns highLimitIndex > lowLimitIndex
        /// </summary>
        /// <param name="accumulativeDistribution"></param>
        /// <param name="random"></param>
        /// <param name="rangesCount"></param>
        /// <returns></returns>
        public static (int lowLimitIndex, int highLimitIndex) GetLimitsIndices(UInt64[] accumulativeDistribution, Random random, int rangesCount)
        {
            UInt64 maxSamples = accumulativeDistribution[^1]; // Последний элемент массиваж
            UInt64 rangeSamples = maxSamples / (UInt64)rangesCount;
            UInt64 lowLimitSamples = (UInt64)(random.NextDouble() * (maxSamples + rangeSamples)) - rangeSamples;
            UInt64 hightLimitSamples = lowLimitSamples + rangeSamples;
            int lowLimitIndex = 0;
            foreach (int i in Enumerable.Range(0, accumulativeDistribution.Length))
            {
                if (lowLimitSamples <= accumulativeDistribution[i])
                {
                    lowLimitIndex = i;
                    break;
                }
            }
            int highLimitIndex = accumulativeDistribution.Length;
            foreach (int i in Enumerable.Range(lowLimitIndex + 1, accumulativeDistribution.Length - lowLimitIndex - 1))
            {
                if (hightLimitSamples <= accumulativeDistribution[i])
                {
                    highLimitIndex = i;
                    break;
                }
            }
            return (lowLimitIndex, highLimitIndex);
        }
    }
}
