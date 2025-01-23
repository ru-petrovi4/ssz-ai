using Ssz.Utils.Serialization;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class StereoInputItem : IOwnedDataSerializable
    {
        public byte Label;

        public byte[] Original_Image = null!;

        public Direction ImageDirection;

        public byte[] LeftEye_Image = null!;

        public byte[] RightEye_Image = null!;

        public DenseMatrix<GradientInPoint> LeftEye_GradientMatrix = null!;

        public DenseMatrix<GradientInPoint> RightEye_GradientMatrix = null!;

        public void SerializeOwnedData(SerializationWriter writer, object? context)        
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(Label);
                writer.WriteArray(Original_Image);
                writer.Write(ImageDirection.XRadians);
                writer.Write(ImageDirection.YRadians);
                writer.WriteArray(LeftEye_Image);
                writer.WriteArray(RightEye_Image);                
                writer.WriteOwnedDataSerializable(LeftEye_GradientMatrix, null);
                writer.WriteOwnedDataSerializable(RightEye_GradientMatrix, null);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        Label = reader.ReadByte();
                        Original_Image = reader.ReadByteArray();
                        ImageDirection.XRadians = reader.ReadSingle();
                        ImageDirection.YRadians = reader.ReadSingle();
                        LeftEye_Image = reader.ReadByteArray();
                        RightEye_Image = reader.ReadByteArray();
                        LeftEye_GradientMatrix = new();
                        reader.ReadOwnedDataSerializable(LeftEye_GradientMatrix, null);
                        RightEye_GradientMatrix = new();
                        reader.ReadOwnedDataSerializable(RightEye_GradientMatrix, null);
                        break;
                }
            }
        }
    }
}
