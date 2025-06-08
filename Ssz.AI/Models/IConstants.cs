namespace Ssz.AI.Models
{
    public interface IConstants
    {
        int RetinaImageWidthPixels { get; }

        int RetinaImageHeightPixels { get; }

        /// <summary>
        ///     Расстояние между детекторами по горизонтали и вертикали              
        /// </summary>
        float RetinaDetectorsDeltaPixels { get; }        

        /// <summary>
        ///     Минимальная чувствительность к модулю градиента
        /// </summary>
        double DetectorMinGradientMagnitude { get; }

        int GeneratedMinGradientMagnitude { get; }

        int GeneratedMaxGradientMagnitude { get; }

        int MagnitudeRangesCount { get; }

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси X
        /// </summary>
        int CortexWidth { get; }

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси Y
        /// </summary>
        int CortexHeight { get; }

        /// <summary>
        ///     Количество детекторов, видимых одной миниколонкой
        /// </summary>
        int MiniColumnVisibleDetectorsCount { get; }

        /// <summary>
        ///     Количество миниколонок в подобласти
        /// </summary>
        int? SubAreaMiniColumnsCount { get; }

        /// <summary>
        ///     Индекс X центра подобласти
        /// </summary>
        int SubAreaCenter_Cx { get; }

        /// <summary>
        ///     Индекс Y центра подобласти
        /// </summary>
        int SubAreaCenter_Cy { get; }

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
        ///     Максимальное расстояние до ближайших миниколонок
        /// </summary>
        float MiniColumnsMaxDistance { get; }

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
        ///     Нулевой уровень косинусного расстояния
        /// </summary>
        float K0 { get; set; }

        /// <summary>
        ///     Порог косинусного расстояния для учета 
        /// </summary>
        float K1 { get; set; }

        /// <summary>
        ///     Косинусное расстояние для пустой колонки
        /// </summary>
        float K2 { get; set; }

        /// <summary>
        ///     Сигмы значимости соседей
        /// </summary>
        float[] K3 { get; set; }

        /// <summary>
        ///     Порог суперактивности
        /// </summary>
        float K4 { get; set; }

        /// <summary>
        ///     Коэффициент для расчета диапазона угла чувствительности детектора
        /// </summary>
        float K5 { get; set; }

        /// <summary>
        ///     Включен ли порог на суперактивность при накоплении воспоминаний
        /// </summary>
        bool SuperactivityThreshold { get; set; }

        float[] PositiveK { get; set; }

        float[] NegativeK { get; set; }
    }
}
