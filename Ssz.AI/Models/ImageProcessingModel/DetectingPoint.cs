using Ssz.AI.Models.Primitives;
using Ssz.Utils;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.ImageProcessingModel;

/// <summary>
///    Детектирующая точка. Набор детекторов разных типов в одной точке.
/// </summary>
public class DetectingPoint : IOwnedDataSerializable
{
    public Retina Retina = null!;

    public int DI;

    public int DJ;

    public double CenterXPixels;

    public double CenterYPixels;

    public SimpleDetector? GradientMagnitude_Detector;

    public SimpleDetector? GradientAngle_Detector;

    public GradientComplexDetector? GradientComplex_Detector;

    public FastList<RetinaPoint> Temp_RetinaPoints = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.WriteOwnedDataSerializable_NullableFixedType(GradientMagnitude_Detector, null);
        writer.WriteOwnedDataSerializable_NullableFixedType(GradientAngle_Detector, null);
        writer.WriteOwnedDataSerializable_NullableFixedType(GradientComplex_Detector, null);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        GradientMagnitude_Detector = reader.ReadOwnedDataSerializable_NullableFixedType<SimpleDetector>(() => new SimpleDetector(FeaturesVector.GradientMagnitude_Index), null);
        GradientAngle_Detector = reader.ReadOwnedDataSerializable_NullableFixedType<SimpleDetector>(() => new SimpleDetector(FeaturesVector.GradientAngle_Index), null);
        GradientComplex_Detector = reader.ReadOwnedDataSerializable_NullableFixedType<GradientComplexDetector>(null);
    }
}
