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
    float MaxGradientMagnitudeExclusive { get; }

    /// <summary>
    ///     Минимальная чувствительность к модулю градиента
    /// </summary>
    float MinGradientMagnitudeInclusive { get; }

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
    ///     Количество детектирующих точек, видимых одной миниколонкой
    /// </summary>
    int MiniColumnVisibleDetectingPointsCount { get; }

    float DetectingPointDeltaPixels => DetectingPointDeltaAngle * RetinaImagePixelSize.Height / RetinaImageAngle;

    float DetectingPointDeltaAngle => FullFieldOfViewDiameter_MiniColumn_Angle / (2.0f * MathF.Sqrt(MiniColumnVisibleDetectingPointsCount / MathF.PI));

    /// <summary>
    ///     Длина хэш-вектора
    /// </summary>
    int HashLength { get; }

    float DetectorRange_MiniColumns { get; }

    float TestGradientAngleDegrees { get; set; }

    float TestGradientMagnitude { get; set; }

    float TestGradientWidthRelative { get; set; }

    float TestGradientPositionRelative { get; set; }
}
