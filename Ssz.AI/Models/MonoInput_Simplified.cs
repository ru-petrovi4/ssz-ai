using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Linq;
using System.Numerics.Tensors;
using static Ssz.AI.Models.Cortex_Simplified;

namespace Ssz.AI.Models
{
    public class MonoInput_Simplified : ISerializableModelObject
    {
        #region construction and destruction

        public MonoInput_Simplified()
        {    
        }

        #endregion

        #region public functions

        /// <summary>
        ///     MonoInputItems.Length = Images.Length + constants.SubAreaMiniColumnsCount
        ///     Last Items - generated ideal Pinwheel
        /// </summary>
        public MonoInputItem[] MonoInputItems = null!;

        public byte[] Labels = null!;
        public byte[][] Images = null!;
        
        public int[] Angle_Big_ToHashIndices = null!;
        public float Angle_BigPoints_Radius;
        public int[] Angle_Small_ToHashIndices = null!;
        public float Angle_SmallPoints_Radius;        

        /// <summary>
        ///     Generates model data after construction.
        /// </summary>
        public void GenerateOwnedData(
            Random random,
            IConstants constants,            
            GradientDistribution? gradientDistribution,
            byte[] labels, 
            byte[][] images)
        {
            Labels = labels;
            Images = images;
            MonoInputItems = new MonoInputItem[images.Length + constants.CalculationsSubAreaRadius_MiniColumns is not null ? constants.CalculationsSubArea_MiniColumns_Count : 0];            
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                MonoInputItem monoInputItem = new();                
                byte[] original_Image = images[i];
                monoInputItem.Label = new Any(labels[i]).ValueAsString(false);
                monoInputItem.Original_Image = original_Image;               

                // Применяем оператор Собеля
                monoInputItem.GradientMatrix = SobelOperator.ApplySobel(original_Image, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);                
                if (gradientDistribution is not null)
                    SobelOperator.CalculateDistribution(monoInputItem.GradientMatrix, gradientDistribution, constants);

                MonoInputItems[i] = monoInputItem;
            }

            Angle_Small_ToHashIndices = new int[constants.Angle_SmallPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Small_ToHashIndices.Length))
            {
                Angle_Small_ToHashIndices[i] = random.Next(constants.HashLength);
            }

            Angle_Big_ToHashIndices = new int[constants.Angle_BigPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Big_ToHashIndices.Length))
            {
                Angle_Big_ToHashIndices[i] = Angle_Small_ToHashIndices[(int)((float)i * Angle_Small_ToHashIndices.Length / Angle_Big_ToHashIndices.Length)];
            }
            Angle_BigPoints_Radius = constants.Angle_BigPoints_Radius;
            Angle_SmallPoints_Radius = constants.Angle_SmallPoints_Radius;
        }

        public void GenerateOwnedData_Simplified(
            Random random,
            IConstants constants,            
            GradientDistribution? gradientDistribution,
            byte[] labels,
            byte[][] images)
        {
            Labels = labels;
            Images = images;
            MonoInputItems = new MonoInputItem[images.Length + constants.CalculationsSubAreaRadius_MiniColumns is not null ? constants.CalculationsSubArea_MiniColumns_Count : 0];
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                MonoInputItem monoInputItem = new();
                byte[] original_Image = images[i];
                monoInputItem.Label = new Any(labels[i]).ValueAsString(false);
                monoInputItem.Original_Image = original_Image;

                // Применяем оператор Собеля
                monoInputItem.GradientMatrix = SobelOperator.ApplySobel_Simplified(original_Image, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
                if (gradientDistribution is not null)
                    SobelOperator.CalculateDistribution(monoInputItem.GradientMatrix, gradientDistribution, constants);

                MonoInputItems[i] = monoInputItem;
            }

            Angle_Small_ToHashIndices = new int[constants.Angle_SmallPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Small_ToHashIndices.Length))
            {
                Angle_Small_ToHashIndices[i] = random.Next(constants.HashLength);
            }

            Angle_Big_ToHashIndices = new int[constants.Angle_BigPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Big_ToHashIndices.Length))
            {
                Angle_Big_ToHashIndices[i] = Angle_Small_ToHashIndices[(int)((float)i * Angle_Small_ToHashIndices.Length / Angle_Big_ToHashIndices.Length)];
            }

            Angle_BigPoints_Radius = constants.Angle_BigPoints_Radius;
            Angle_SmallPoints_Radius = constants.Angle_SmallPoints_Radius;
        }

        public void GenerateOwnedData_Simplified2(
            Random random,
            IConstants constants,            
            GradientDistribution? gradientDistribution,
            byte[] labels,
            byte[][] images)
        {
            Labels = labels;
            Images = images;
            if (constants.CalculationsSubAreaRadius_MiniColumns is not null)
                MonoInputItems = new MonoInputItem[images.Length + constants.CalculationsSubArea_MiniColumns_Count + 10]; // С запасом
            else
                MonoInputItems = new MonoInputItem[images.Length];
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Вычисляем магнитуду и угол градиента
                double magnitude = constants.GeneratedMinGradientMagnitude +
                    (constants.GeneratedMaxGradientMagnitude - constants.GeneratedMinGradientMagnitude) * random.NextDouble();
                // [-pi, pi]
                double angle = -Math.PI + random.NextDouble() * 2 * Math.PI; // Угол в радианах

                MonoInputItem monoInputItem = new();
                byte[] original_Image = images[i];
                monoInputItem.Label = $"Maginitude: {(int)magnitude}; Angle: {(int)MathHelper.RadiansToDegrees((float)angle)}";
                monoInputItem.Original_Image = original_Image;

                // Применяем оператор Собеля
                monoInputItem.GradientMatrix = SobelOperator.ApplySobel_Simplified2(original_Image, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels, magnitude, angle);
                if (gradientDistribution is not null)
                    SobelOperator.CalculateDistribution(monoInputItem.GradientMatrix, gradientDistribution, constants);

                MonoInputItems[i] = monoInputItem;
            }

            Angle_Small_ToHashIndices = new int[constants.Angle_SmallPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Small_ToHashIndices.Length))
            {
                Angle_Small_ToHashIndices[i] = random.Next(constants.HashLength);
            }

            Angle_Big_ToHashIndices = new int[constants.Angle_BigPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Big_ToHashIndices.Length))
            {
                Angle_Big_ToHashIndices[i] = Angle_Small_ToHashIndices[(int)((float)i * Angle_Small_ToHashIndices.Length / Angle_Big_ToHashIndices.Length)];
            }

            Angle_BigPoints_Radius = constants.Angle_BigPoints_Radius;
            Angle_SmallPoints_Radius = constants.Angle_SmallPoints_Radius;
        }

        public void GenerateOwnedData_Simplified_WithAngle(
            Random random,
            IConstants constants,            
            GradientDistribution? gradientDistribution,
            byte[] labels,
            byte[][] images)
        {
            Labels = labels;
            Images = images;
            MonoInputItems = new MonoInputItem[images.Length + constants.CalculationsSubAreaRadius_MiniColumns is not null ? constants.CalculationsSubArea_MiniColumns_Count : 0];
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Вычисляем магнитуду и угол градиента
                double magnitude = constants.GeneratedMaxGradientMagnitude * random.NextDouble();
                // [-pi, pi]
                double angle = -Math.PI + random.NextDouble() * 2 * Math.PI; // Угол в радианах

                MonoInputItem monoInputItem = new();
                byte[] original_Image = images[i];
                monoInputItem.Label = $"Maginitude: {(int)magnitude}; Angle: {(int)MathHelper.RadiansToDegrees((float)angle)}";
                monoInputItem.Original_Image = original_Image;

                // Применяем оператор Собеля
                monoInputItem.GradientMatrix = SobelOperator.ApplySobel_Simplified2(original_Image, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels, magnitude, angle);
                if (gradientDistribution is not null)
                    SobelOperator.CalculateDistribution(monoInputItem.GradientMatrix, gradientDistribution, constants);

                MonoInputItems[i] = monoInputItem;
            }

            Angle_Small_ToHashIndices = new int[constants.Angle_SmallPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Small_ToHashIndices.Length))
            {
                Angle_Small_ToHashIndices[i] = random.Next(constants.HashLength);
            }

            Angle_Big_ToHashIndices = new int[constants.Angle_BigPoints_Count];
            foreach (int i in Enumerable.Range(0, Angle_Big_ToHashIndices.Length))
            {
                Angle_Big_ToHashIndices[i] = Angle_Small_ToHashIndices[(int)((float)i * Angle_Small_ToHashIndices.Length / Angle_Big_ToHashIndices.Length)];
            }

            Angle_BigPoints_Radius = constants.Angle_BigPoints_Radius;
            Angle_SmallPoints_Radius = constants.Angle_SmallPoints_Radius;
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

        public void AddValueHash(float angleNormalized, float[] temp_Hash)
        {
            Hash.ValueToHash(
                    angleNormalized,
                    Angle_Big_ToHashIndices,
                    Angle_Small_ToHashIndices,
                    bigRadius: Angle_BigPoints_Radius,
                    smallRadius: Angle_SmallPoints_Radius,
                    temp_Hash);
        }

        #endregion        
    }

    public struct MonoInputItem : IOwnedDataSerializable
    {
        public string Label;

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
            Label = reader.ReadString();
            Original_Image = reader.ReadByteArray();
            GradientMatrix = new();
            reader.ReadOwnedDataSerializable(GradientMatrix, null);
        }
    }
}
