using OpenCvSharp;

namespace Ssz.AI.Models
{
    public class Detector
    {
        public double CenterX { get; }
        public double CenterY { get; }
        public double Width { get; }
        public double AngleRange { get; }
        public double GradientMagnitudeRange { get; }

        public Detector(double centerX, double centerY, double width, double angleRange, double gradientMagnitudeRange)
        {
            CenterX = centerX;
            CenterY = centerY;
            Width = width;
            AngleRange = angleRange;
            GradientMagnitudeRange = gradientMagnitudeRange;
        }

        public bool IsActivated(Mat image)
        {
            var interpolatedValue = image.At<double>((int)CenterX, (int)CenterY);
            return AngleRange <= interpolatedValue && interpolatedValue <= GradientMagnitudeRange;
        }
    }
}
