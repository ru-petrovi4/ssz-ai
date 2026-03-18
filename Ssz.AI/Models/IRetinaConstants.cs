using Avalonia;
using System;

namespace Ssz.AI.Models;

/// <summary>
///     Все углы по умолчанию в радианах.
/// </summary>
public interface IRetinaConstants
{
    PixelSize RetinaImagePixelSize { get; }

    float RetinaImageAngle { get; }    

    /// <summary>
    ///     Оценочный максимальный градиент.
    /// </summary>
    int MaxGradientMagnitudeExclusive { get; }

    /// <summary>
    ///     Минимальная чувствительность к модулю градиента
    /// </summary>
    double MinGradientMagnitudeInclusive { get; }

    /// <summary>
    ///     For internal calculations of Detectors densities.
    /// </summary>
    float GradientMagnitudeDelta { get; }

    /// <summary>
    ///     For internal calculations of Detectors densities.
    /// </summary>
    float GradientAngleDegreeDelta { get; }

    Vector3DFloat PhysicalImageCenter { get; }

    Size2DFloat PhysicalImageSize { get; }

    float DistanceBetweenEyes { get; }

    float RetinaPointDeltaPixels { get; }

    float DetectorFieldOfViewRadiusPixels { get; }

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

    float RetinaDetectorsDeltaPixels => RetinaDetectorsDeltaAngle * RetinaImagePixelSize.Height / RetinaImageAngle;

    float RetinaDetectorsDeltaAngle => FullFieldOfViewDiameter_MiniColumn_Angle / (2.0f * MathF.Sqrt(MiniColumnVisibleDetectorsCount / MathF.PI));

    /// <summary>
    ///     Длина хэш-вектора
    /// </summary>
    int HashLength { get; }

    float DetectorRange_MiniColumns { get; }
}
