using Avalonia;
using System;

namespace Ssz.AI.Models
{
    public interface IConstants : IMiniColumnsActivityConstants
    {
        PixelSize RetinaImagePixelSize { get; set; }

        /// <summary>
        ///     Расстояние между детекторами по горизонтали и вертикали              
        /// </summary>
        float RetinaDetectorsDeltaPixels { get; set; }

        /// <summary>
        ///     Количество детекторов, видимых одной миниколонкой
        /// </summary>
        int MiniColumnVisibleDetectorsCount { get; }

        /// <summary>
        ///     Минимальная чувствительность к модулю градиента
        /// </summary>
        double DetectorMinGradientMagnitude { get; }

        int GeneratedMinGradientMagnitude { get; }

        int GeneratedMaxGradientMagnitude { get; }

        int MagnitudeRangesCount { get; }

        /// <summary>
        ///     Количество миниколонок в подобласти
        /// </summary>
        float? CalculationsSubAreaRadius_MiniColumns { get; }

        int CalculationsSubArea_MiniColumns_Count => (int)(MathF.PI * CalculationsSubAreaRadius_MiniColumns!.Value * CalculationsSubAreaRadius_MiniColumns.Value);

        /// <summary>
        ///     Индекс X центра подобласти
        /// </summary>
        int CalculationsSubAreaCenter_Cx { get; }

        /// <summary>
        ///     Индекс Y центра подобласти
        /// </summary>
        int CalculationsSubAreaCenter_Cy { get; }

        /// <summary>
        ///     Примерный радиус гиперколонки (измеренный в миниколонках).
        /// </summary>
        float HyperColumnSupposedRadius_MiniColumns { get; }

        float HyperColumnSupposedRadius_ForMemorySaving_MiniColumns { get; }

        /// <summary>
        ///     Количество гиперколнок в рецептивном поле миниколонки.
        /// </summary>
        float DetectorsField_HyperColumns { get; }

        /// <summary>
        ///     Длина хэш-вектора
        /// </summary>
        int HashLength { get; }

        /// <summary>
        ///     Длина короткого хэш-вектора
        /// </summary>
        int ShortHashLength { get; }

        /// <summary>
        ///     Количество бит в коротком хэш-векторе
        /// </summary>
        int ShortHashBitsCount { get; }

        /// <summary>
        ///     Минимальное число бит в хэше, что бы быть сохраненным в память
        /// </summary>
        int MinBitsInHashForMemory { get; }        

        /// <summary>
        ///     Верхний предел количества воспоминаний (для кэширования)
        /// </summary>
        int MemoriesMaxCount { get; }

        /// <summary>
        ///     Порог для кластеризации воспоминаний
        /// </summary>
        float MemoryClustersThreshold { get; }

        int Angle_SmallPoints_Count { get; }

        float Angle_SmallPoints_Radius { get; }

        int Angle_BigPoints_Count { get; }

        float Angle_BigPoints_Radius { get; }        

        /// <summary>
        ///     Порог косинусного расстояния для учета 
        /// </summary>
        float K1 { get; set; }        

        /// <summary>
        ///     Сигмы значимости соседей
        /// </summary>
        float[] K3 { get; set; }        

        /// <summary>
        ///     Коэффициент для расчета диапазона угла чувствительности детектора
        /// </summary>
        float K5 { get; set; }        
    }
}
