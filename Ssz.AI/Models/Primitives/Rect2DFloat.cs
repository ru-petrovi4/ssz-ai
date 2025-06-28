using System;
using System.Globalization;
using System.Numerics;

namespace Ssz.AI.Models;

/// <summary>
/// Defines a rectangle.
/// </summary>
public readonly struct Rect2DFloat : IEquatable<Rect2DFloat>
{
    /// <summary>
    /// The X position.
    /// </summary>
    private readonly float _x;

    /// <summary>
    /// The Y position.
    /// </summary>
    private readonly float _y;

    /// <summary>
    /// The width.
    /// </summary>
    private readonly float _width;

    /// <summary>
    /// The height.
    /// </summary>
    private readonly float _height;

    /// <summary>
    /// Initializes a new instance of the <see cref="Rect2DFloat"/> structure.
    /// </summary>
    /// <param name="x">The X position.</param>
    /// <param name="y">The Y position.</param>
    /// <param name="width">The width.</param>
    /// <param name="height">The height.</param>
    public Rect2DFloat(float x, float y, float width, float height)
    {
        _x = x;
        _y = y;
        _width = width;
        _height = height;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Rect2DFloat"/> structure.
    /// </summary>
    /// <param name="size">The size of the rectangle.</param>
    public Rect2DFloat(Size2DFloat size)
    {
        _x = 0;
        _y = 0;
        _width = size.Width;
        _height = size.Height;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Rect2DFloat"/> structure.
    /// </summary>
    /// <param name="position">The position of the rectangle.</param>
    /// <param name="size">The size of the rectangle.</param>
    public Rect2DFloat(Point2DFloat position, Size2DFloat size)
    {
        _x = position.X;
        _y = position.Y;
        _width = size.Width;
        _height = size.Height;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Rect2DFloat"/> structure.
    /// </summary>
    /// <param name="topLeft">The top left position of the rectangle.</param>
    /// <param name="bottomRight">The bottom right position of the rectangle.</param>
    public Rect2DFloat(Point2DFloat topLeft, Point2DFloat bottomRight)
    {
        _x = topLeft.X;
        _y = topLeft.Y;
        _width = bottomRight.X - topLeft.X;
        _height = bottomRight.Y - topLeft.Y;
    }

    /// <summary>
    /// Gets the X position.
    /// </summary>
    public float X => _x;

    /// <summary>
    /// Gets the Y position.
    /// </summary>
    public float Y => _y;

    /// <summary>
    /// Gets the width.
    /// </summary>
    public float Width => _width;

    /// <summary>
    /// Gets the height.
    /// </summary>
    public float Height => _height;

    /// <summary>
    /// Gets the position of the rectangle.
    /// </summary>
    public Point2DFloat Position => new Point2DFloat(_x, _y);

    /// <summary>
    /// Gets the size of the rectangle.
    /// </summary>
    public Size2DFloat Size => new Size2DFloat(_width, _height);

    /// <summary>
    /// Gets the right position of the rectangle.
    /// </summary>
    public float Right => _x + _width;

    /// <summary>
    /// Gets the bottom position of the rectangle.
    /// </summary>
    public float Bottom => _y + _height;

    /// <summary>
    /// Gets the left position.
    /// </summary>
    public float Left => _x;

    /// <summary>
    /// Gets the top position.
    /// </summary>
    public float Top => _y;

    /// <summary>
    /// Gets the top left point of the rectangle.
    /// </summary>
    public Point2DFloat TopLeft => new Point2DFloat(_x, _y);

    /// <summary>
    /// Gets the top right point of the rectangle.
    /// </summary>
    public Point2DFloat TopRight => new Point2DFloat(Right, _y);

    /// <summary>
    /// Gets the bottom left point of the rectangle.
    /// </summary>
    public Point2DFloat BottomLeft => new Point2DFloat(_x, Bottom);

    /// <summary>
    /// Gets the bottom right point of the rectangle.
    /// </summary>
    public Point2DFloat BottomRight => new Point2DFloat(Right, Bottom);

    /// <summary>
    /// Gets the center point of the rectangle.
    /// </summary>
    public Point2DFloat Center => new Point2DFloat(_x + (_width / 2), _y + (_height / 2));

    /// <summary>
    /// Checks for equality between two <see cref="Rect2DFloat"/>s.
    /// </summary>
    /// <param name="left">The first rect.</param>
    /// <param name="right">The second rect.</param>
    /// <returns>True if the rects are equal; otherwise false.</returns>
    public static bool operator ==(Rect2DFloat left, Rect2DFloat right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Checks for inequality between two <see cref="Rect2DFloat"/>s.
    /// </summary>
    /// <param name="left">The first rect.</param>
    /// <param name="right">The second rect.</param>
    /// <returns>True if the rects are unequal; otherwise false.</returns>
    public static bool operator !=(Rect2DFloat left, Rect2DFloat right)
    {
        return !(left == right);
    }

    /// <summary>
    /// Multiplies a rectangle by a scaling vector.
    /// </summary>
    /// <param name="rect">The rectangle.</param>
    /// <param name="scale">The vector scale.</param>
    /// <returns>The scaled rectangle.</returns>
    public static Rect2DFloat operator *(Rect2DFloat rect, Point2DFloat scale)
    {
        return new Rect2DFloat(
            rect.X * scale.X,
            rect.Y * scale.Y,
            rect.Width * scale.X,
            rect.Height * scale.Y);
    }

    /// <summary>
    /// Multiplies a rectangle by a scale.
    /// </summary>
    /// <param name="rect">The rectangle.</param>
    /// <param name="scale">The scale.</param>
    /// <returns>The scaled rectangle.</returns>
    public static Rect2DFloat operator *(Rect2DFloat rect, float scale)
    {
        return new Rect2DFloat(
            rect.X * scale,
            rect.Y * scale,
            rect.Width * scale,
            rect.Height * scale);
    }

    /// <summary>
    /// Divides a rectangle by a vector.
    /// </summary>
    /// <param name="rect">The rectangle.</param>
    /// <param name="scale">The vector scale.</param>
    /// <returns>The scaled rectangle.</returns>
    public static Rect2DFloat operator /(Rect2DFloat rect, Point2DFloat scale)
    {
        return new Rect2DFloat(
            rect.X / scale.X,
            rect.Y / scale.Y,
            rect.Width / scale.X,
            rect.Height / scale.Y);
    }

    /// <summary>
    /// Determines whether a point is in the bounds of the rectangle.
    /// </summary>
    /// <param name="p">The point.</param>
    /// <returns>true if the point is in the bounds of the rectangle; otherwise false.</returns>
    public bool Contains(Point2DFloat p)
    {
        return p.X >= _x && p.X <= _x + _width &&
               p.Y >= _y && p.Y <= _y + _height;
    }

    /// <summary>
    /// Determines whether a point is in the bounds of the rectangle, exclusive of the
    /// rectangle's bottom/right edge.
    /// </summary>
    /// <param name="p">The point.</param>
    /// <returns>true if the point is in the bounds of the rectangle; otherwise false.</returns>    
    public bool ContainsExclusive(Point2DFloat p)
    {
        return p.X >= _x && p.X < _x + _width &&
               p.Y >= _y && p.Y < _y + _height;
    }

    /// <summary>
    /// Determines whether the rectangle fully contains another rectangle.
    /// </summary>
    /// <param name="r">The rectangle.</param>
    /// <returns>true if the rectangle is fully contained; otherwise false.</returns>
    public bool Contains(Rect2DFloat r)
    {
        return Contains(r.TopLeft) && Contains(r.BottomRight);
    }

    /// <summary>
    /// Centers another rectangle in this rectangle.
    /// </summary>
    /// <param name="rect">The rectangle to center.</param>
    /// <returns>The centered rectangle.</returns>
    public Rect2DFloat CenterRectFloat(Rect2DFloat rect)
    {
        return new Rect2DFloat(
            _x + ((_width - rect._width) / 2),
            _y + ((_height - rect._height) / 2),
            rect._width,
            rect._height);
    }

    /// <summary>
    /// Inflates the rectangle.
    /// </summary>
    /// <param name="thickness">The thickness to be subtracted for each side of the rectangle.</param>
    /// <returns>The inflated rectangle.</returns>
    public Rect2DFloat Inflate(float thickness)
    {
        return Inflate(new Thickness2DFloat(thickness));
    }

    /// <summary>
    /// Inflates the rectangle.
    /// </summary>
    /// <param name="thickness">The thickness to be subtracted for each side of the rectangle.</param>
    /// <returns>The inflated rectangle.</returns>
    public Rect2DFloat Inflate(Thickness2DFloat thickness)
    {
        return new Rect2DFloat(
            new Point2DFloat(_x - thickness.Left, _y - thickness.Top),
            Size.Inflate(thickness));
    }

    /// <summary>
    /// Deflates the rectangle.
    /// </summary>
    /// <param name="thickness">The thickness to be subtracted for each side of the rectangle.</param>
    /// <returns>The deflated rectangle.</returns>
    public Rect2DFloat Deflate(float thickness)
    {
        return Deflate(new Thickness2DFloat(thickness));
    }

    /// <summary>
    /// Deflates the rectangle by a <see cref="Thickness2DFloat"/>.
    /// </summary>
    /// <param name="thickness">The thickness to be subtracted for each side of the rectangle.</param>
    /// <returns>The deflated rectangle.</returns>
    public Rect2DFloat Deflate(Thickness2DFloat thickness)
    {
        return new Rect2DFloat(
            new Point2DFloat(_x + thickness.Left, _y + thickness.Top),
            Size.Deflate(thickness));
    }

    /// <summary>
    /// Returns a boolean indicating whether the rect is equal to the other given rect.
    /// </summary>
    /// <param name="other">The other rect to test equality against.</param>
    /// <returns>True if this rect is equal to other; False otherwise.</returns>
    public bool Equals(Rect2DFloat other)
    {
        // ReSharper disable CompareOfFloatsByEqualityOperator
        return _x == other._x &&
               _y == other._y &&
               _width == other._width &&
               _height == other._height;
        // ReSharper enable CompareOfFloatsByEqualityOperator
    }

    /// <summary>
    /// Returns a boolean indicating whether the given object is equal to this rectangle.
    /// </summary>
    /// <param name="obj">The object to compare against.</param>
    /// <returns>True if the object is equal to this rectangle; false otherwise.</returns>
    public override bool Equals(object? obj) => obj is Rect2DFloat other && Equals(other);

    /// <summary>
    /// Returns the hash code for this instance.
    /// </summary>
    /// <returns>The hash code.</returns>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 23) + X.GetHashCode();
            hash = (hash * 23) + Y.GetHashCode();
            hash = (hash * 23) + Width.GetHashCode();
            hash = (hash * 23) + Height.GetHashCode();
            return hash;
        }
    }

    /// <summary>
    /// Gets the intersection of two rectangles.
    /// </summary>
    /// <param name="rect">The other rectangle.</param>
    /// <returns>The intersection.</returns>
    public Rect2DFloat Intersect(Rect2DFloat rect)
    {
        var newLeft = (rect.X > X) ? rect.X : X;
        var newTop = (rect.Y > Y) ? rect.Y : Y;
        var newRight = (rect.Right < Right) ? rect.Right : Right;
        var newBottom = (rect.Bottom < Bottom) ? rect.Bottom : Bottom;

        if ((newRight > newLeft) && (newBottom > newTop))
        {
            return new Rect2DFloat(newLeft, newTop, newRight - newLeft, newBottom - newTop);
        }
        else
        {
            return default;
        }
    }

    /// <summary>
    /// Determines whether a rectangle intersects with this rectangle.
    /// </summary>
    /// <param name="rect">The other rectangle.</param>
    /// <returns>
    /// True if the specified rectangle intersects with this one; otherwise false.
    /// </returns>
    public bool Intersects(Rect2DFloat rect)
    {
        return (rect.X < Right) && (X < rect.Right) && (rect.Y < Bottom) && (Y < rect.Bottom);
    }

    ///// <summary>
    ///// Returns the axis-aligned bounding box of a transformed rectangle.
    ///// </summary>
    ///// <param name="matrix">The transform.</param>
    ///// <returns>The bounding box</returns>
    //public Rect2DFloat TransformToAABB(Matrix matrix)
    //{
    //    ReadOnlySpan<Vector2DFloat> points = stackalloc Vector2DFloat[4]
    //    {
    //            TopLeft.Transform(matrix),
    //            TopRight.Transform(matrix),
    //            BottomRight.Transform(matrix),
    //            BottomLeft.Transform(matrix)
    //        };

    //    var left = float.MaxValue;
    //    var right = float.MinValue;
    //    var top = float.MaxValue;
    //    var bottom = float.MinValue;

    //    foreach (var p in points)
    //    {
    //        if (p.X < left) left = p.X;
    //        if (p.X > right) right = p.X;
    //        if (p.Y < top) top = p.Y;
    //        if (p.Y > bottom) bottom = p.Y;
    //    }

    //    return new Rect2DFloat(new Vector2DFloat(left, top), new Vector2DFloat(right, bottom));
    //}

    internal Rect2DFloat TransformToAABB(Matrix4x4 matrix)
    {
        ReadOnlySpan<Point2DFloat> points = stackalloc Point2DFloat[4]
        {
                TopLeft.Transform(matrix),
                TopRight.Transform(matrix),
                BottomRight.Transform(matrix),
                BottomLeft.Transform(matrix)
            };

        var left = float.MaxValue;
        var right = float.MinValue;
        var top = float.MaxValue;
        var bottom = float.MinValue;

        foreach (var p in points)
        {
            if (p.X < left) left = p.X;
            if (p.X > right) right = p.X;
            if (p.Y < top) top = p.Y;
            if (p.Y > bottom) bottom = p.Y;
        }

        return new Rect2DFloat(new Point2DFloat(left, top), new Point2DFloat(right, bottom));
    }

    /// <summary>
    /// Translates the rectangle by an offset.
    /// </summary>
    /// <param name="offset">The offset.</param>
    /// <returns>The translated rectangle.</returns>
    public Rect2DFloat Translate(Point2DFloat offset)
    {
        return new Rect2DFloat(Position + offset, Size);
    }

    /// <summary>
    /// Normalizes the rectangle so both the <see cref="Width"/> and <see 
    /// cref="Height"/> are positive, without changing the location of the rectangle
    /// </summary>
    /// <returns>Normalized RectFloat</returns>
    /// <remarks>
    /// Empty rect will be return when RectFloat contains invalid values. Like NaN.
    /// </remarks>
    public Rect2DFloat Normalize()
    {
        Rect2DFloat rect = this;

        if (float.IsNaN(rect.Right) || float.IsNaN(rect.Bottom) ||
            float.IsNaN(rect.X) || float.IsNaN(rect.Y) ||
            float.IsNaN(Height) || float.IsNaN(Width))
        {
            return default;
        }

        if (rect.Width < 0)
        {
            var x = X + Width;
            var width = X - x;

            rect = rect.WithX(x).WithWidth(width);
        }

        if (rect.Height < 0)
        {
            var y = Y + Height;
            var height = Y - y;

            rect = rect.WithY(y).WithHeight(height);
        }

        return rect;
    }

    /// <summary>
    /// Gets the union of two rectangles.
    /// </summary>
    /// <param name="rect">The other rectangle.</param>
    /// <returns>The union.</returns>
    public Rect2DFloat Union(Rect2DFloat rect)
    {
        if (Width == 0 && Height == 0)
        {
            return rect;
        }
        else if (rect.Width == 0 && rect.Height == 0)
        {
            return this;
        }
        else
        {
            var x1 = Math.Min(X, rect.X);
            var x2 = Math.Max(Right, rect.Right);
            var y1 = Math.Min(Y, rect.Y);
            var y2 = Math.Max(Bottom, rect.Bottom);

            return new Rect2DFloat(new Point2DFloat(x1, y1), new Point2DFloat(x2, y2));
        }
    }

    internal static Rect2DFloat? Union(Rect2DFloat? left, Rect2DFloat? right)
    {
        if (left == null)
            return right;
        if (right == null)
            return left;
        return left.Value.Union(right.Value);
    }

    /// <summary>
    /// Returns a new <see cref="Rect2DFloat"/> with the specified X position.
    /// </summary>
    /// <param name="x">The x position.</param>
    /// <returns>The new <see cref="Rect2DFloat"/>.</returns>
    public Rect2DFloat WithX(float x)
    {
        return new Rect2DFloat(x, _y, _width, _height);
    }

    /// <summary>
    /// Returns a new <see cref="Rect2DFloat"/> with the specified Y position.
    /// </summary>
    /// <param name="y">The y position.</param>
    /// <returns>The new <see cref="Rect2DFloat"/>.</returns>
    public Rect2DFloat WithY(float y)
    {
        return new Rect2DFloat(_x, y, _width, _height);
    }

    /// <summary>
    /// Returns a new <see cref="Rect2DFloat"/> with the specified width.
    /// </summary>
    /// <param name="width">The width.</param>
    /// <returns>The new <see cref="Rect2DFloat"/>.</returns>
    public Rect2DFloat WithWidth(float width)
    {
        return new Rect2DFloat(_x, _y, width, _height);
    }

    /// <summary>
    /// Returns a new <see cref="Rect2DFloat"/> with the specified height.
    /// </summary>
    /// <param name="height">The height.</param>
    /// <returns>The new <see cref="Rect2DFloat"/>.</returns>
    public Rect2DFloat WithHeight(float height)
    {
        return new Rect2DFloat(_x, _y, _width, height);
    }

    /// <summary>
    /// Returns the string representation of the rectangle.
    /// </summary>
    /// <returns>The string representation of the rectangle.</returns>
    public override string ToString()
    {
        return string.Format(
            CultureInfo.InvariantCulture,
            "{0}, {1}, {2}, {3}",
            _x,
            _y,
            _width,
            _height);
    }    

    /// <summary>
    /// This method should be used internally to check for the rect emptiness
    /// Once we add support for WPF-like empty rects, there will be an actual implementation
    /// For now it's internal to keep some loud community members happy about the API being pretty 
    /// </summary>
    internal bool IsEmpty() => this == default;
}
