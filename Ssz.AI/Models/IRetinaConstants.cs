using Avalonia;
using System;

namespace Ssz.AI.Models;

/// <summary>
///     Все углы по умолчанию в радианах.
/// </summary>
public interface IRetinaConstants
{
    PixelSize RetinaImagePixelSize { get; set; }

    float RetinaImageAngle { get; set; }    

    /// <summary>
    ///     Оценочный максимальный градиент.
    /// </summary>
    int MaxGradientMagnitudeExclusive { get; }

    /// <summary>
    ///     Минимальная чувствительность к модулю градиента
    /// </summary>
    double DetectorMinGradientMagnitudeInclusive { get; }

    float GradientMagnitudeDelta { get; }

    float GradientAngleDegreeDelta { get; }

    Vector3DFloat PhysicalImageCenter { get; }

    Size2DFloat PhysicalImageSize { get; }

    float DistanceBetweenEyes { get; }

    /// <summary>
    ///     Примерный радиус гиперколонки (измеренный в миниколонках).
    /// </summary>
    int HyperColumnDefinedRadius_MiniColumns { get; }

    /// <summary>
    ///     Полное поле зрения (измеренное в миниколонках).
    ///     <para>При смечщении на такое число миниколонок, поле зрения смещается на 100%.</para>
    /// </summary>
    int FullFieldOfView_MiniColumns { get; }

    float FullFieldOfViewDiameter_MiniColumn_Angle { get; }

    /// <summary>
    ///     Количество детекторов, видимых одной миниколонкой
    /// </summary>
    int MiniColumnVisibleDetectorsCount { get; }

    float RetinaDetectorsDeltaPixels => (RetinaImagePixelSize.Height * FullFieldOfViewDiameter_MiniColumn_Angle) / (RetinaImageAngle * MathF.Sqrt(MiniColumnVisibleDetectorsCount / MathF.PI));

    float RetinaDetectorsDeltaAngle => FullFieldOfViewDiameter_MiniColumn_Angle / MathF.Sqrt(MiniColumnVisibleDetectorsCount / MathF.PI);

    /// <summary>
    ///     Длина хэш-вектора
    /// </summary>
    int HashLength { get; }
}
