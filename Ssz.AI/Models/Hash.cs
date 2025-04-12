using System;

namespace Ssz.AI.Models;

public static class Hash
{
    /// <summary>
    ///     Hash is NOT cleared.
    /// </summary>
    /// <param name="value">[0.0..1.0)</param>
    /// <param name="value_ToHashIndices"></param>
    /// <param name="bigDelta"></param>
    /// <param name="smallRadius"></param>
    /// <param name="bigRadius"></param>
    /// <param name="hash"></param>
    public static void ValueToHash(
        float value, 
        int[] value_BigToHashIndices,
        int[] value_SmallToHashIndices,
        float bigRadius,
        float smallRadius,         
        float[] hash)
    {
        // Hash from Bigs
        float bigDelta = 1.0f / value_BigToHashIndices.Length;
        for (float v = value - bigRadius; v < value + bigRadius; v += bigDelta)
        {
            if (v >= 0.0f && v < 1.0f)
            {
                int i = (int)(v / bigDelta);
                if (i < value_BigToHashIndices.Length)
                    hash[value_BigToHashIndices[i]] = 1.0f;
            }
        }

        // Hash from Smalls
        float smallDelta = 1.0f / value_SmallToHashIndices.Length;
        for (float v = value - smallRadius; v < value + smallRadius; v += smallDelta)
        {
            if (v >= 0.0f && v < 1.0f)
            {
                int i = (int)(v / smallDelta);
                if (i < value_SmallToHashIndices.Length)
                    hash[value_SmallToHashIndices[i]] = 1.0f;
            }
        }
    }
}
