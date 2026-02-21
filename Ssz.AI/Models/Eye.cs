using Avalonia;

namespace Ssz.AI.Models
{
    public class Eye
    {
        /// <summary>
        ///     Upper left corner or retina relative to pupil
        /// </summary>
        public Vector3DFloat Pupil = null!;

        public Retina Retina = null!;

        public float RetinaUpperLeftXAngle;

        public float RetinaUpperLeftYAngle;

        public float RetinaBottomRightXAngle;

        public float RetinaBottomRightYAngle;
    }
}
