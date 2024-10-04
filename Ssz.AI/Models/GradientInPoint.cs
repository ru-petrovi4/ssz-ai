namespace Ssz.AI.Models
{
    public struct GradientInPoint
    {
        public int GradX;
        public int GradY;
        /// <summary>
        ///     1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))
        /// </summary>
        public double Magnitude;
        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double Angle;
    }
}
