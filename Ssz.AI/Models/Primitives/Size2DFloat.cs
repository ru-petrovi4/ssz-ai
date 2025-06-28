using System;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models;

public readonly struct Size2DFloat : IEquatable<Size2DFloat>
{
    /// <summary>
    /// A size representing infinity.
    /// </summary>
    public static readonly Size2DFloat Infinity = new Size2DFloat(float.PositiveInfinity, float.PositiveInfinity);

    /// <summary>
    /// The width.
    /// </summary>
    private readonly float _width;

    /// <summary>
    /// The height.
    /// </summary>
    private readonly float _height;

    /// <summary>
    /// Initializes a new instance of the <see cref="Size2DFloat"/> structure.
    /// </summary>
    /// <param name="width">The width.</param>
    /// <param name="height">The height.</param>
    public Size2DFloat(float width, float height)
    {
        _width = width;
        _height = height;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Size2DFloat"/> structure.
    /// </summary>
    /// <param name="vector2">The vector to take values from.</param>
    public Size2DFloat(System.Numerics.Vector2 vector2) : this(vector2.X, vector2.Y)
    {

    }

    /// <summary>
    /// Gets the aspect ratio of the size.
    /// </summary>
    public float AspectRatio => _width / _height;

    /// <summary>
    /// Gets the width.
    /// </summary>
    public float Width => _width;

    /// <summary>
    /// Gets the height.
    /// </summary>
    public float Height => _height;

    /// <summary>
    /// Checks for equality between two <see cref="Size2DFloat"/>s.
    /// </summary>
    /// <param name="left">The first size.</param>
    /// <param name="right">The second size.</param>
    /// <returns>True if the sizes are equal; otherwise false.</returns>
    public static bool operator ==(Size2DFloat left, Size2DFloat right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Checks for inequality between two <see cref="Size2DFloat"/>s.
    /// </summary>
    /// <param name="left">The first size.</param>
    /// <param name="right">The second size.</param>
    /// <returns>True if the sizes are unequal; otherwise false.</returns>
    public static bool operator !=(Size2DFloat left, Size2DFloat right)
    {
        return !(left == right);
    }

    /// <summary>
    /// Scales a size.
    /// </summary>
    /// <param name="size">The size</param>
    /// <param name="scale">The scaling factor.</param>
    /// <returns>The scaled size.</returns>
    public static Size2DFloat operator *(Size2DFloat size, Point2DFloat scale)
    {
        return new Size2DFloat(size._width * scale.X, size._height * scale.Y);
    }

    /// <summary>
    /// Scales a size.
    /// </summary>
    /// <param name="size">The size</param>
    /// <param name="scale">The scaling factor.</param>
    /// <returns>The scaled size.</returns>
    public static Size2DFloat operator /(Size2DFloat size, Point2DFloat scale)
    {
        return new Size2DFloat(size._width / scale.X, size._height / scale.Y);
    }

    /// <summary>
    /// Divides a size by another size to produce a scaling factor.
    /// </summary>
    /// <param name="left">The first size</param>
    /// <param name="right">The second size.</param>
    /// <returns>The scaled size.</returns>
    public static Point2DFloat operator /(Size2DFloat left, Size2DFloat right)
    {
        return new Point2DFloat(left._width / right._width, left._height / right._height);
    }

    /// <summary>
    /// Scales a size.
    /// </summary>
    /// <param name="size">The size</param>
    /// <param name="scale">The scaling factor.</param>
    /// <returns>The scaled size.</returns>
    public static Size2DFloat operator *(Size2DFloat size, float scale)
    {
        return new Size2DFloat(size._width * scale, size._height * scale);
    }

    /// <summary>
    /// Scales a size.
    /// </summary>
    /// <param name="size">The size</param>
    /// <param name="scale">The scaling factor.</param>
    /// <returns>The scaled size.</returns>
    public static Size2DFloat operator /(Size2DFloat size, float scale)
    {
        return new Size2DFloat(size._width / scale, size._height / scale);
    }

    public static Size2DFloat operator +(Size2DFloat size, Size2DFloat toAdd)
    {
        return new Size2DFloat(size._width + toAdd._width, size._height + toAdd._height);
    }

    public static Size2DFloat operator -(Size2DFloat size, Size2DFloat toSubtract)
    {
        return new Size2DFloat(size._width - toSubtract._width, size._height - toSubtract._height);
    }

    ///// <summary>
    ///// Parses a <see cref="Size2DFloat"/> string.
    ///// </summary>
    ///// <param name="s">The string.</param>
    ///// <returns>The <see cref="Size2DFloat"/>.</returns>
    //public static Size2DFloat Parse(string s)
    //{
    //    using (var tokenizer = new SpanStringTokenizer(s, CultureInfo.InvariantCulture, exceptionMessage: "Invalid Size2DFloat."))
    //    {
    //        return new Size2DFloat(
    //            tokenizer.ReadDouble(),
    //            tokenizer.ReadDouble());
    //    }
    //}

    /// <summary>
    /// Constrains the size.
    /// </summary>
    /// <param name="constraint">The size to constrain to.</param>
    /// <returns>The constrained size.</returns>
    public Size2DFloat Constrain(Size2DFloat constraint)
    {
        return new Size2DFloat(
            Math.Min(_width, constraint._width),
            Math.Min(_height, constraint._height));
    }

    /// <summary>
    /// Deflates the size by a <see cref="Thickness2DFloat"/>.
    /// </summary>
    /// <param name="thickness">The thickness.</param>
    /// <returns>The deflated size.</returns>
    /// <remarks>The deflated size cannot be less than 0.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Size2DFloat Deflate(Thickness2DFloat thickness)
    {
        var width = _width - thickness.Left - thickness.Right;
        if (width < 0)
            width = 0;

        var height = _height - thickness.Top - thickness.Bottom;
        if (height < 0)
            height = 0;

        return new Size2DFloat(width, height);
    }

    /// <summary>
    /// Returns a boolean indicating whether the size is equal to the other given size (bitwise).
    /// </summary>
    /// <param name="other">The other size to test equality against.</param>
    /// <returns>True if this size is equal to other; False otherwise.</returns>
    public bool Equals(Size2DFloat other)
    {
        // ReSharper disable CompareOfFloatsByEqualityOperator
        return _width == other._width &&
               _height == other._height;
        // ReSharper enable CompareOfFloatsByEqualityOperator
    }

    ///// <summary>
    ///// Returns a boolean indicating whether the size is equal to the other given size (numerically).
    ///// </summary>
    ///// <param name="other">The other size to test equality against.</param>
    ///// <returns>True if this size is equal to other; False otherwise.</returns>
    //public bool NearlyEquals(Size2DFloat other)
    //{
    //    return MathUtilities.AreClose(_width, other._width) &&
    //           MathUtilities.AreClose(_height, other._height);
    //}

    /// <summary>
    /// Checks for equality between a size and an object.
    /// </summary>
    /// <param name="obj">The object.</param>
    /// <returns>
    /// True if <paramref name="obj"/> is a size that equals the current size.
    /// </returns>
    public override bool Equals(object? obj) => obj is Size2DFloat other && Equals(other);

    /// <summary>
    /// Returns a hash code for a <see cref="Size2DFloat"/>.
    /// </summary>
    /// <returns>The hash code.</returns>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 23) + Width.GetHashCode();
            hash = (hash * 23) + Height.GetHashCode();
            return hash;
        }
    }

    /// <summary>
    /// Inflates the size by a <see cref="Thickness2DFloat"/>.
    /// </summary>
    /// <param name="thickness">The thickness.</param>
    /// <returns>The inflated size.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Size2DFloat Inflate(Thickness2DFloat thickness)
    {
        return new Size2DFloat(
            _width + thickness.Left + thickness.Right,
            _height + thickness.Top + thickness.Bottom);
    }

    /// <summary>
    /// Returns a new <see cref="Size2DFloat"/> with the same height and the specified width.
    /// </summary>
    /// <param name="width">The width.</param>
    /// <returns>The new <see cref="Size2DFloat"/>.</returns>
    public Size2DFloat WithWidth(float width)
    {
        return new Size2DFloat(width, _height);
    }

    /// <summary>
    /// Returns a new <see cref="Size2DFloat"/> with the same width and the specified height.
    /// </summary>
    /// <param name="height">The height.</param>
    /// <returns>The new <see cref="Size2DFloat"/>.</returns>
    public Size2DFloat WithHeight(float height)
    {
        return new Size2DFloat(_width, height);
    }

    /// <summary>
    /// Returns the string representation of the size.
    /// </summary>
    /// <returns>The string representation of the size.</returns>
    public override string ToString()
    {
        return string.Format(CultureInfo.InvariantCulture, "{0}, {1}", _width, _height);
    }

    /// <summary>
    /// Deconstructs the size into its Width and Height values.
    /// </summary>
    /// <param name="width">The width.</param>
    /// <param name="height">The height.</param>
    public void Deconstruct(out float width, out float height)
    {
        width = this._width;
        height = this._height;
    }
}
