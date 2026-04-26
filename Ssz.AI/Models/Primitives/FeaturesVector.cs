using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.Primitives;

[InlineArray(Size)]
public struct FeaturesVector
{
    public const int Size = 2;

    private float _element0; // остальное генерируется компилятором

    public const int GradientMagnitude_Index = 0;
    public const int GradientAngle_Index = 1;
}
