using System;

namespace Ssz.AI.Models;

public readonly struct Thickness2DFloat : IEquatable<Thickness2DFloat>
{
    /// <summary>
    /// The thickness on the left.
    /// </summary>
    private readonly float _left;

    /// <summary>
    /// The thickness on the top.
    /// </summary>
    private readonly float _top;

    /// <summary>
    /// The thickness on the right.
    /// </summary>
    private readonly float _right;

    /// <summary>
    /// The thickness on the bottom.
    /// </summary>
    private readonly float _bottom;

    /// <summary>
    /// Initializes a new instance of the <see cref="Thickness2DFloat"/> structure.
    /// </summary>
    /// <param name="uniformLength">The length that should be applied to all sides.</param>
    public Thickness2DFloat(float uniformLength)
    {
        _left = _top = _right = _bottom = uniformLength;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Thickness2DFloat"/> structure.
    /// </summary>
    /// <param name="horizontal">The thickness on the left and right.</param>
    /// <param name="vertical">The thickness on the top and bottom.</param>
    public Thickness2DFloat(float horizontal, float vertical)
    {
        _left = _right = horizontal;
        _top = _bottom = vertical;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Thickness2DFloat"/> structure.
    /// </summary>
    /// <param name="left">The thickness on the left.</param>
    /// <param name="top">The thickness on the top.</param>
    /// <param name="right">The thickness on the right.</param>
    /// <param name="bottom">The thickness on the bottom.</param>
    public Thickness2DFloat(float left, float top, float right, float bottom)
    {
        _left = left;
        _top = top;
        _right = right;
        _bottom = bottom;
    }

    /// <summary>
    /// Gets the thickness on the left.
    /// </summary>
    public float Left => _left;

    /// <summary>
    /// Gets the thickness on the top.
    /// </summary>
    public float Top => _top;

    /// <summary>
    /// Gets the thickness on the right.
    /// </summary>
    public float Right => _right;

    /// <summary>
    /// Gets the thickness on the bottom.
    /// </summary>
    public float Bottom => _bottom;

    /// <summary>
    /// Gets a value indicating whether all sides are equal.
    /// </summary>
    public bool IsUniform => Left.Equals(Right) && Top.Equals(Bottom) && Right.Equals(Bottom);

    /// <summary>
    /// Compares two Thickness2DFloates.
    /// </summary>
    /// <param name="a">The first thickness.</param>
    /// <param name="b">The second thickness.</param>
    /// <returns>The equality.</returns>
    public static bool operator ==(Thickness2DFloat a, Thickness2DFloat b)
    {
        return a.Equals(b);
    }

    /// <summary>
    /// Compares two Thickness2DFloates.
    /// </summary>
    /// <param name="a">The first thickness.</param>
    /// <param name="b">The second thickness.</param>
    /// <returns>The inequality.</returns>
    public static bool operator !=(Thickness2DFloat a, Thickness2DFloat b)
    {
        return !a.Equals(b);
    }

    /// <summary>
    /// Adds two Thickness2DFloates.
    /// </summary>
    /// <param name="a">The first thickness.</param>
    /// <param name="b">The second thickness.</param>
    /// <returns>The equality.</returns>
    public static Thickness2DFloat operator +(Thickness2DFloat a, Thickness2DFloat b)
    {
        return new Thickness2DFloat(
            a.Left + b.Left,
            a.Top + b.Top,
            a.Right + b.Right,
            a.Bottom + b.Bottom);
    }

    /// <summary>
    /// Subtracts two Thickness2DFloates.
    /// </summary>
    /// <param name="a">The first thickness.</param>
    /// <param name="b">The second thickness.</param>
    /// <returns>The equality.</returns>
    public static Thickness2DFloat operator -(Thickness2DFloat a, Thickness2DFloat b)
    {
        return new Thickness2DFloat(
            a.Left - b.Left,
            a.Top - b.Top,
            a.Right - b.Right,
            a.Bottom - b.Bottom);
    }

    /// <summary>
    /// Multiplies a Thickness2DFloat to a scalar.
    /// </summary>
    /// <param name="a">The thickness.</param>
    /// <param name="b">The scalar.</param>
    /// <returns>The equality.</returns>
    public static Thickness2DFloat operator *(Thickness2DFloat a, float b)
    {
        return new Thickness2DFloat(
            a.Left * b,
            a.Top * b,
            a.Right * b,
            a.Bottom * b);
    }

    /// <summary>
    /// Adds a Thickness2DFloat to a Size2DFloat.
    /// </summary>
    /// <param name="size">The size.</param>
    /// <param name="thickness">The thickness.</param>
    /// <returns>The equality.</returns>
    public static Size2DFloat operator +(Size2DFloat size, Thickness2DFloat thickness)
    {
        return new Size2DFloat(
            size.Width + thickness.Left + thickness.Right,
            size.Height + thickness.Top + thickness.Bottom);
    }

    /// <summary>
    /// Subtracts a Thickness2DFloat from a Size2DFloat.
    /// </summary>
    /// <param name="size">The size.</param>
    /// <param name="thickness">The thickness.</param>
    /// <returns>The equality.</returns>
    public static Size2DFloat operator -(Size2DFloat size, Thickness2DFloat thickness)
    {
        return new Size2DFloat(
            size.Width - (thickness.Left + thickness.Right),
            size.Height - (thickness.Top + thickness.Bottom));
    }

    ///// <summary>
    ///// Parses a <see cref="Thickness2DFloat"/> string.
    ///// </summary>
    ///// <param name="s">The string.</param>
    ///// <returns>The <see cref="Thickness2DFloat"/>.</returns>
    //public static Thickness2DFloat Parse(string s)
    //{
    //    const string exceptionMessage = "Invalid Thickness2DFloat.";

    //    using (var tokenizer = new SpanStringTokenizer(s, CultureInfo.InvariantCulture, exceptionMessage))
    //    {
    //        if (tokenizer.TryReadDouble(out var a))
    //        {
    //            if (tokenizer.TryReadDouble(out var b))
    //            {
    //                if (tokenizer.TryReadDouble(out var c))
    //                {
    //                    return new Thickness2DFloat(a, b, c, tokenizer.ReadDouble());
    //                }

    //                return new Thickness2DFloat(a, b);
    //            }

    //            return new Thickness2DFloat(a);
    //        }

    //        throw new FormatException(exceptionMessage);
    //    }
    //}

    /// <summary>
    /// Returns a boolean indicating whether the thickness is equal to the other given point.
    /// </summary>
    /// <param name="other">The other thickness to test equality against.</param>
    /// <returns>True if this thickness is equal to other; False otherwise.</returns>
    public bool Equals(Thickness2DFloat other)
    {
        // ReSharper disable CompareOfFloatsByEqualityOperator
        return _left == other._left &&
               _top == other._top &&
               _right == other._right &&
               _bottom == other._bottom;
        // ReSharper restore CompareOfFloatsByEqualityOperator
    }

    /// <summary>
    /// Checks for equality between a thickness and an object.
    /// </summary>
    /// <param name="obj">The object.</param>
    /// <returns>
    /// True if <paramref name="obj"/> is a size that equals the current size.
    /// </returns>
    public override bool Equals(object? obj) => obj is Thickness2DFloat other && Equals(other);

    /// <summary>
    /// Returns a hash code for a <see cref="Thickness2DFloat"/>.
    /// </summary>
    /// <returns>The hash code.</returns>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 23) + Left.GetHashCode();
            hash = (hash * 23) + Top.GetHashCode();
            hash = (hash * 23) + Right.GetHashCode();
            hash = (hash * 23) + Bottom.GetHashCode();
            return hash;
        }
    }

    /// <summary>
    /// Returns the string representation of the thickness.
    /// </summary>
    /// <returns>The string representation of the thickness.</returns>
    public override string ToString()
    {
        return FormattableString.Invariant($"{_left},{_top},{_right},{_bottom}");
    }

    /// <summary>
    /// Deconstructor the thickness into its left, top, right and bottom thickness values.
    /// </summary>
    /// <param name="left">The thickness on the left.</param>
    /// <param name="top">The thickness on the top.</param>
    /// <param name="right">The thickness on the right.</param>
    /// <param name="bottom">The thickness on the bottom.</param>
    public void Deconstruct(out float left, out float top, out float right, out float bottom)
    {
        left = this._left;
        top = this._top;
        right = this._right;
        bottom = this._bottom;
    }
}