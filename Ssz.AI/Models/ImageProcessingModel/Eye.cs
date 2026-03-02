using Avalonia;
using System;

namespace Ssz.AI.Models.ImageProcessingModel;

public class Eye
{
    /// <summary>
    ///     Upper left corner or retina relative to pupil
    /// </summary>
    public Vector3DFloat Pupil = null!;

    public Retina Retina = null!;

    public bool IsRightEye;

    public float RetinaUpperLeftXAbsoluteAngle;

    public float RetinaUpperLeftYAbsoluteAngle;

    public float RetinaBottomRightXAbsoluteAngle;

    public float RetinaBottomRightYAbsoluteAngle;    
}
