using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System;
using System.Linq;
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Models
{
    public class MonoInput : ISerializableModelObject
    {
        #region construction and destruction

        public MonoInput()
        {    
        }

        #endregion

        #region public functions

        public MonoInputItem[] MonoInputItems = null!;

        /// <summary>
        ///     Generates model data after construction.
        /// </summary>
        public void GenerateOwnedData(
            ICortexConstants constants,
            GradientDistribution? gradientDistribution,
            byte[] labels, 
            byte[][] images)
        {
            MonoInputItems = new MonoInputItem[images.Length];            
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                MonoInputItem monoInputItem = new();                
                byte[] original_Image = images[i];
                monoInputItem.Label = labels[i];
                monoInputItem.Original_Image = original_Image;               

                // Применяем оператор Собеля
                monoInputItem.GradientMatrix = SobelOperator.ApplySobel(original_Image, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);                
                if (gradientDistribution is not null)
                    SobelOperator.CalculateDistribution(monoInputItem.GradientMatrix, gradientDistribution, constants);

                MonoInputItems[i] = monoInputItem;
            }
        }

        /// <summary>
        ///     Prepares for calculation after DeserializeOwnedData or GenerateOwnedData
        /// </summary>
        public void Prepare()
        {
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArrayOfOwnedDataSerializable(MonoInputItems, null);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        MonoInputItems = reader.ReadArrayOfOwnedDataSerializable(() => default(MonoInputItem), null);
                        break;                    
                }
            }
        }

        #endregion
    }

    public struct MonoInputItem : IOwnedDataSerializable
    {
        public byte Label;

        public byte[] Original_Image;        

        public DenseMatrix<GradientInPoint> GradientMatrix;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            writer.Write(Label);
            writer.WriteArray(Original_Image);
            writer.WriteOwnedDataSerializable(GradientMatrix, null);
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            Label = reader.ReadByte();
            Original_Image = reader.ReadByteArray();
            GradientMatrix = new();
            reader.ReadOwnedDataSerializable(GradientMatrix, null);
        }
    }
}
