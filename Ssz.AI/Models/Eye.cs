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

        public float RetinaUpperLeftXRadians;

        public float RetinaUpperLeftYRadians;

        public float RetinaBottomRightXRadians;

        public float RetinaBottomRightYRadians;
    }
}
