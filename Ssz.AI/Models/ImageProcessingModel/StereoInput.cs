using Avalonia;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System;
using System.Drawing;
using System.Linq;

namespace Ssz.AI.Models.ImageProcessingModel;

public class StereoInput : ISerializableModelObject
{
    #region construction and destruction

    public StereoInput()
    { 
    }

    #endregion

    #region public functions

    public StereoInputSample[] StereoInputSamples = null!;

    /// <summary>
    ///     Generates model data after construction.
    /// </summary>
    public void GenerateOwnedData(
        PixelSize inputImagesSize,
        Random initializationRandom,
        IRetinaConstants constants,
        GradientDistribution leftEye_GradientDistribution,
        GradientDistribution rightEye_GradientDistribution,
        byte[] inputImagesLabels, 
        byte[][] inputImageDatas,            
        Eye leftEye,
        Eye rightEye)
    {
        StereoInputSamples = new StereoInputSample[inputImageDatas.Length];            
        for (int index = 0; index < inputImageDatas.Length; index += 1)
        {
            StereoInputSample stereoInputSample = new();
            StereoInputSamples[index] = stereoInputSample;
            stereoInputSample.Index = index;
            byte[] inputImageData = inputImageDatas[index];
            stereoInputSample.Label = inputImagesLabels[index];
            stereoInputSample.InputImageData = inputImageData;
            stereoInputSample.ImageNormalDirection = new Direction();
            stereoInputSample.ImageNormalDirection.XRadians = -MathF.PI / 4 + initializationRandom.NextSingle() * MathF.PI / 2;
            stereoInputSample.ImageNormalDirection.YRadians = -MathF.PI / 4 + initializationRandom.NextSingle() * MathF.PI / 2;

            stereoInputSample.LeftRetinaImageData = GetRetinaImageData(constants, stereoInputSample.InputImageData, inputImagesSize, stereoInputSample.ImageNormalDirection, leftEye);
            stereoInputSample.RightRetinaImageData = GetRetinaImageData(constants, stereoInputSample.InputImageData, inputImagesSize, stereoInputSample.ImageNormalDirection, rightEye);

            // Применяем оператор Собеля
            stereoInputSample.LeftEye_GradientMatrix = SobelOperator.ApplySobel(stereoInputSample.LeftRetinaImageData, constants.RetinaImagePixelSize.Width, constants.RetinaImagePixelSize.Height);
            
            //SobelOperator.CalculateDistribution(stereoInputSample.LeftEye_GradientMatrix, leftEye_GradientDistribution, constants);
            Array.Fill(leftEye_GradientDistribution.MagnitudeData, 1u, 0, 1200);
            
            stereoInputSample.RightEye_GradientMatrix = SobelOperator.ApplySobel(stereoInputSample.RightRetinaImageData, constants.RetinaImagePixelSize.Width, constants.RetinaImagePixelSize.Height);
            
            //SobelOperator.CalculateDistribution(stereoInputSample.RightEye_GradientMatrix, rightEye_GradientDistribution, constants);
            Array.Fill(rightEye_GradientDistribution.MagnitudeData, 1u, 0, 1200);
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
            writer.WriteArrayOfOwnedDataSerializable(StereoInputSamples, null);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    StereoInputSamples = reader.ReadArrayOfOwnedDataSerializable(() => new StereoInputSample(), null);
                    break;                    
            }
        }
    }

    public static byte[] GetRetinaImageData(IRetinaConstants constants, byte[] inputImageData, PixelSize inputImageSize, Direction imageNormalDirection, Eye eye)
    {
        float widthRadians = eye.RetinaBottomRightXAbsoluteAngle - eye.RetinaUpperLeftXAbsoluteAngle;
        float heightRadians = eye.RetinaBottomRightYAbsoluteAngle - eye.RetinaUpperLeftYAbsoluteAngle;

        byte[] retinaImageData = new byte[constants.RetinaImagePixelSize.Width * constants.RetinaImagePixelSize.Height];
        for (int y = 0; y < constants.RetinaImagePixelSize.Height; y += 1)
        {
            for (int x = 0; x < constants.RetinaImagePixelSize.Width; x += 1)
            {
                Direction currentDirection = new();
                currentDirection.XRadians = eye.RetinaUpperLeftXAbsoluteAngle + widthRadians * x / constants.RetinaImagePixelSize.Width;
                currentDirection.YRadians = eye.RetinaUpperLeftYAbsoluteAngle + heightRadians * y / constants.RetinaImagePixelSize.Height;
                (float centerX, float centerY) = GetPointOnImage(constants, eye.Pupil, currentDirection, imageNormalDirection, inputImageSize);
                // Значение пикселя из массива байтов
                byte pixelValue = BitmapHelper.GetInterpolatedValue(inputImageData, inputImageSize, centerX, centerY);

                retinaImageData[x + y * constants.RetinaImagePixelSize.Width] = pixelValue;
            }
        }
        return retinaImageData;            
    }

    #endregion                

    private static (float centerX, float centerY) GetPointOnImage(IRetinaConstants constants, Vector3DFloat pupil, Direction currentDirection, Direction imageNormalDirection, PixelSize inputImageSize)
    {
        // Входные данные
        // Координаты точки A и углы линии Л
        float Ax = pupil.X, Ay = pupil.Y, Az = pupil.Z;
        float lineAngleXZ = currentDirection.XRadians; // угол в плоскости XZ
        float lineAngleYZ = currentDirection.YRadians; // угол в плоскости YZ

        // Координаты точки B и углы нормали плоскости П
        float Bx = constants.PhysicalImageCenter.X, By = constants.PhysicalImageCenter.Y, Bz = constants.PhysicalImageCenter.Z;
        float normalAngleXZ = imageNormalDirection.XRadians; // угол в плоскости XZ
        float normalAngleYZ = imageNormalDirection.YRadians; // угол в плоскости YZ

        // Направляющий вектор линии Л
        float lineDirX = MathF.Sin(lineAngleXZ);
        float lineDirY = MathF.Sin(lineAngleYZ);
        float lineDirZ = MathF.Cos(lineAngleXZ) * MathF.Cos(lineAngleYZ);

        // Направляющий вектор нормали плоскости П
        float normalX = MathF.Sin(normalAngleXZ);
        float normalY = MathF.Sin(normalAngleYZ);
        float normalZ = MathF.Cos(normalAngleXZ) * MathF.Cos(normalAngleYZ);

        // Вектор точки A -> точки B
        float ABx = Bx - Ax;
        float ABy = By - Ay;
        float ABz = Bz - Az;

        // Скалярное произведение направляющего вектора линии и нормали
        float dotProduct = lineDirX * normalX + lineDirY * normalY + lineDirZ * normalZ;

        // Проверка на параллельность линии и плоскости
        if (MathF.Abs(dotProduct) < 1e-6)
        {
            Console.WriteLine("Линия и плоскость параллельны или лежат в одной плоскости.");
            return (float.NaN, float.NaN);
        }

        // Параметр t для точки пересечения
        float t = (ABx * normalX + ABy * normalY + ABz * normalZ) / dotProduct;

        // Координаты точки пересечения в трехмерном пространстве
        float intersectX = Ax + t * lineDirX;
        float intersectY = Ay + t * lineDirY;
        float intersectZ = Az + t * lineDirZ;

        // Вектор из точки B до точки пересечения
        float BIntersectX = intersectX - Bx;
        float BIntersectY = intersectY - By;
        float BIntersectZ = intersectZ - Bz;

        // Преобразование в двумерные координаты на плоскости П
        // Ось X плоскости
        float planeXDirX = MathF.Cos(normalAngleXZ);
        float planeXDirY = 0.0f;
        float planeXDirZ = -MathF.Sin(normalAngleXZ);

        // Ось Y плоскости
        (float planeYDirX, float planeYDirY, float planeYDirZ) = VectorProduct(normalX, normalY, normalZ, planeXDirX, planeXDirY, planeXDirZ);

        //float l = MathF.Sqrt(planeYDirX * planeYDirX + planeYDirY * planeYDirY + planeYDirZ * planeYDirZ);
        //planeYDirX /= l;
        //planeYDirY /= l;
        //planeYDirZ /= l;

        // Координаты в системе плоскости
        float planeCoordX = BIntersectX * planeXDirX + BIntersectY * planeXDirY + BIntersectZ * planeXDirZ;
        float planeCoordY = BIntersectX * planeYDirX + BIntersectY * planeYDirY + BIntersectZ * planeYDirZ;

        planeCoordX -= constants.PhysicalImageCenter.X - constants.PhysicalImageSize.Width / 2;
        planeCoordY -= constants.PhysicalImageCenter.Y - constants.PhysicalImageSize.Height / 2;

        planeCoordX /= constants.PhysicalImageSize.Width / inputImageSize.Width;
        planeCoordY /= constants.PhysicalImageSize.Height / inputImageSize.Height;

        return (planeCoordX, planeCoordY);
    }

    private static (float, float, float) VectorProduct(float ax, float ay, float az, float bx, float by, float bz)
    {
        float planeYDirX = ay * bz - az * by;
        float planeYDirY = az * bx - ax * bz;
        float planeYDirZ = ax * by - ay * bx;
        return (planeYDirX, planeYDirY, planeYDirZ);
    }
}

public class StereoInputSample : IOwnedDataSerializable
{
    public int Index;

    public byte Label;

    public byte[] InputImageData = null!;

    public Direction ImageNormalDirection;

    public byte[] LeftRetinaImageData = null!;

    public byte[] RightRetinaImageData = null!;

    public DenseMatrix<GradientInPoint> LeftEye_GradientMatrix = null!;

    public DenseMatrix<GradientInPoint> RightEye_GradientMatrix = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.Write(Index);
            writer.Write(Label);
            writer.WriteArray(InputImageData);
            writer.Write(ImageNormalDirection.XRadians);
            writer.Write(ImageNormalDirection.YRadians);
            writer.WriteArray(LeftRetinaImageData);
            writer.WriteArray(RightRetinaImageData);
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
                    Index = reader.ReadInt32();
                    Label = reader.ReadByte();
                    InputImageData = reader.ReadByteArray();
                    ImageNormalDirection.XRadians = reader.ReadSingle();
                    ImageNormalDirection.YRadians = reader.ReadSingle();
                    LeftRetinaImageData = reader.ReadByteArray();
                    RightRetinaImageData = reader.ReadByteArray();
                    LeftEye_GradientMatrix = new();
                    reader.ReadOwnedDataSerializable(LeftEye_GradientMatrix, null);
                    RightEye_GradientMatrix = new();
                    reader.ReadOwnedDataSerializable(RightEye_GradientMatrix, null);
                    break;
            }
        }
    }
}

public class SampleVisualisation
{
    public Image? FullImage;

    public Image? Image;

    public Image? GradientImage;

    public Image? DetectorsActivationImage;
}