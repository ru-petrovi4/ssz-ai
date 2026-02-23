using Avalonia;
using System;

namespace Ssz.AI.Models;

/// <summary>
///     Все углы по умолчанию в радианах.
/// </summary>
public interface IRetinaConstants
{
    PixelSize RetinaImagePixelSize { get; set; }

    float RetinaImageVerticalAngle { get; set; }    

    /// <summary>
    ///     Оценочный максимальный градиент.
    /// </summary>
    int MaxGradientMagnitudeExclusive { get; }

    /// <summary>
    ///     Минимальная чувствительность к модулю градиента
    /// </summary>
    double DetectorMinGradientMagnitudeInclusive { get; }

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

    float MiniColumnFieldOfViewDiameter_Angle { get; }

    /// <summary>
    ///     Количество детекторов, видимых одной миниколонкой
    /// </summary>
    int MiniColumnVisibleDetectorsCount { get; }

    float RetinaDetectorsDeltaPixels => (RetinaImagePixelSize.Height * MiniColumnFieldOfViewDiameter_Angle) / (RetinaImageVerticalAngle * MathF.Sqrt(MiniColumnVisibleDetectorsCount / MathF.PI));

    /// <summary>
    ///     Длина хэш-вектора
    /// </summary>
    int HashLength { get; }
}
