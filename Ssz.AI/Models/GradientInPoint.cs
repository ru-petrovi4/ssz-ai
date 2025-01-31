using Ssz.Utils.Serialization;

namespace Ssz.AI.Models
{
    public struct GradientInPoint : IOwnedDataSerializable
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

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(GradX);
                writer.Write(GradY);
                writer.Write(Magnitude);
                writer.Write(Angle);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        GradX = reader.ReadInt32();
                        GradY = reader.ReadInt32();
                        Magnitude = reader.ReadDouble();
                        Angle = reader.ReadDouble();
                        break;
                }
            }
        }
    }
}
