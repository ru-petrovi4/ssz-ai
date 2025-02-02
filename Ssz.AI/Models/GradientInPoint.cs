using Ssz.Utils.Serialization;

namespace Ssz.AI.Models
{
    public struct GradientInPoint : IOwnedDataSerializable
    {
        public double GradX;
        public double GradY;
        /// <summary>
        ///     1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))
        /// </summary>
        public double Magnitude;
        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double Angle;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            writer.Write(GradX);
            writer.Write(GradY);
            writer.Write(Magnitude);
            writer.Write(Angle);
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            GradX = reader.ReadDouble();
            GradY = reader.ReadDouble();
            Magnitude = reader.ReadDouble();
            Angle = reader.ReadDouble();
        }
    }
}
