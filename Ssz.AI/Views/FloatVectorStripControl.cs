using System;
using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace Ssz.AI.Views;

public class FloatVectorStripControl : Control
{
    public static readonly StyledProperty<IReadOnlyList<float>?> ValuesProperty =
        AvaloniaProperty.Register<FloatVectorStripControl, IReadOnlyList<float>?>(nameof(Values));

    public static readonly StyledProperty<double> StripHeightProperty =
        AvaloniaProperty.Register<FloatVectorStripControl, double>(nameof(StripHeight), 24);

    public static readonly StyledProperty<double> CellSpacingProperty =
        AvaloniaProperty.Register<FloatVectorStripControl, double>(nameof(CellSpacing), 1);

    static FloatVectorStripControl()
    {
        AffectsRender<FloatVectorStripControl>(ValuesProperty, StripHeightProperty, CellSpacingProperty);
        AffectsMeasure<FloatVectorStripControl>(ValuesProperty, StripHeightProperty, CellSpacingProperty);
    }

    public IReadOnlyList<float>? Values
    {
        get => GetValue(ValuesProperty);
        set => SetValue(ValuesProperty, value);
    }

    public double StripHeight
    {
        get => GetValue(StripHeightProperty);
        set => SetValue(StripHeightProperty, value);
    }

    public double CellSpacing
    {
        get => GetValue(CellSpacingProperty);
        set => SetValue(CellSpacingProperty, value);
    }

    protected override Size MeasureOverride(Size availableSize)
    {
        var count = Values?.Count ?? 0;
        var side = Math.Max(1, StripHeight);
        var spacing = Math.Max(0, CellSpacing);

        var width = count == 0 ? 0 : count * side + Math.Max(0, count - 1) * spacing;
        return new Size(width, side);
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);

        var values = Values;
        if (values is null || values.Count == 0)
            return;

        var side = Math.Max(1, StripHeight);
        var spacing = Math.Max(0, CellSpacing);

        for (int i = 0; i < values.Count; i++)
        {
            var gray = ToGrayByte(values[i]);
            var brush = new SolidColorBrush(Color.FromRgb(gray, gray, gray));

            var x = i * (side + spacing);
            var rect = new Rect(x, 0, side, side);

            context.FillRectangle(brush, rect);
        }
    }

    private static byte ToGrayByte(float value)
    {
        var clamped = Math.Clamp(value, 0f, 2f);
        var normalized = clamped / 2f;
        return (byte)Math.Round(normalized * 255f);
    }
}
