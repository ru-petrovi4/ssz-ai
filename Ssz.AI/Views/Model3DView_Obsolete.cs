using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.OpenGL;
using Avalonia.OpenGL.Controls;
using Avalonia.Platform;
using Avalonia.Threading;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Ssz.AI.Models;
using System;
using System.Collections.Generic;
using System.Numerics;


namespace Ssz.AI.Views
{

    //private void DrawPoints()
    //{
    //    Model3DScene? model3DScene = Data;
    //    if (model3DScene is null || model3DScene.Point3DWithColorWithColorArray is null)
    //        return;

    //    GL.Begin(PrimitiveType.Points);

    //    for (int i = 0; i < model3DScene.Point3DWithColorWithColorArray.Length; i++)
    //    {
    //        var point3DWithColor = model3DScene.Point3DWithColorWithColorArray[i];

    //        GL.Color3(point3DWithColor.Color.R, point3DWithColor.Color.G, point3DWithColor.Color.B);
    //        GL.Vertex3(point3DWithColor.X, point3DWithColor.Y, point3DWithColor.Z);
    //    }

    //    GL.End();
    //}

    public class Model3DView_Obsolete : OpenGlControlBase
    {
        #region public functions

        public static readonly AvaloniaProperty<Model3DScene?> DataProperty = AvaloniaProperty.Register<Model3DView, Model3DScene?>(
            nameof(Data));

        public Model3DScene? Data
        {
            get => GetValue(DataProperty) as Model3DScene;
            set => SetValue(DataProperty, value);
        }

        #endregion

        private GL? _gl;
        private uint _vao, _vbo, _shaderProgram;
        private Point3DWithColor[]? _points;
        private float _rotationX, _rotationY;
        private float _zoom = 5.0f;
        private Vector2D<float> _lastMousePos;

        public Model3DView_Obsolete()
        {            
            this.PointerPressed += OnPointerPressed;
            this.PointerMoved += OnPointerMoved;
            this.PointerWheelChanged += OnPointerWheelChanged;
        }

        protected override void OnOpenGlInit(GlInterface gl)
        {
            base.OnOpenGlInit(gl);

            _gl = GL.GetApi(gl.GetProcAddress);            

            // Vertex Shader
            string vertexShaderSource = """
            #version 330 core
            layout (location = 0) in vec3 aPosition;
            layout (location = 1) in vec4 aColor;
            out vec4 vertexColor;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            void main()
            {
                gl_Position = projection * view * model * vec4(aPosition, 1.0);
                vertexColor = aColor;
                gl_PointSize = 5.0;
            }
            """;

            // Fragment Shader
            string fragmentShaderSource = """
            #version 330 core
            in vec4 vertexColor;
            out vec4 FragColor;
            void main()
            {
                FragColor = vertexColor;
            }
            """;

            // Компиляция шейдеров
            uint vertexShader = _gl.CreateShader(ShaderType.VertexShader);
            _gl.ShaderSource(vertexShader, vertexShaderSource);
            _gl.CompileShader(vertexShader);

            uint fragmentShader = _gl.CreateShader(ShaderType.FragmentShader);
            _gl.ShaderSource(fragmentShader, fragmentShaderSource);
            _gl.CompileShader(fragmentShader);

            _shaderProgram = _gl.CreateProgram();
            _gl.AttachShader(_shaderProgram, vertexShader);
            _gl.AttachShader(_shaderProgram, fragmentShader);
            _gl.LinkProgram(_shaderProgram);

            _gl.DeleteShader(vertexShader);
            _gl.DeleteShader(fragmentShader);

            // Настройка буферов
            _vao = _gl.GenVertexArray();
            _vbo = _gl.GenBuffer();

            _gl.BindVertexArray(_vao);
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);                       

            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 7 * sizeof(float), 0);

            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, 7 * sizeof(float), 3 * sizeof(float));
        }

        protected override void OnOpenGlDeinit(GlInterface gl)
        {
            
            base.OnOpenGlDeinit(gl);
        }

        protected override unsafe void OnOpenGlRender(GlInterface gl, int fb)
        {
            Render();

            Dispatcher.UIThread.Post(RequestNextFrameRendering, DispatcherPriority.Background);
        }

        public void Render()
        {
            _points = Data?.Point3DWithColorArray;
            if (_gl is null || _points is null) return;

            _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            _gl.Enable(EnableCap.DepthTest);
            _gl.Enable(EnableCap.ProgramPointSize);

            _gl.UseProgram(_shaderProgram);


            float[] vertexData = new float[1000 * 7];
            for (int i = 0; i < _points.Length && i < 1000; i++)
            {
                int offset = i * 7;
                vertexData[offset] = _points[i].Position.X;
                vertexData[offset + 1] = _points[i].Position.Y;
                vertexData[offset + 2] = _points[i].Position.Z;
                vertexData[offset + 3] = _points[i].Color.X;
                vertexData[offset + 4] = _points[i].Color.Y;
                vertexData[offset + 5] = _points[i].Color.Z;
                vertexData[offset + 6] = _points[i].Color.W;
            }
            unsafe
            {
                fixed (void* d = vertexData)
                {
                    _gl.BufferData(BufferTargetARB.ArrayBuffer, (uint)(vertexData.Length * sizeof(float)), d, BufferUsageARB.StaticDraw);
                }
            }


            // Матрицы трансформации
            var model = Matrix4X4<float>.Identity * Matrix4X4.CreateRotationX(_rotationX) * Matrix4X4.CreateRotationY(_rotationY);
            var view = Matrix4X4.CreateLookAt(new Vector3D<float>(0, 0, _zoom), Vector3D<float>.Zero, Vector3D<float>.UnitY);
            var projection = Matrix4X4.CreatePerspectiveFieldOfView(MathF.PI / 4, (float)Bounds.Width / (float)Bounds.Height, 0.1f, 100.0f);

            unsafe
            {
                _gl.UniformMatrix4(_gl.GetUniformLocation(_shaderProgram, "model"), 1, false, (float*)&model);
                _gl.UniformMatrix4(_gl.GetUniformLocation(_shaderProgram, "view"), 1, false, (float*)&view);
                _gl.UniformMatrix4(_gl.GetUniformLocation(_shaderProgram, "projection"), 1, false, (float*)&projection);
            }            

            _gl.BindVertexArray(_vao);
            _gl.DrawArrays(PrimitiveType.Points, 0, (uint)_points.Length);
        }

        private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
        {
            _lastMousePos = new Vector2D<float>((float)e.GetPosition(this).X, (float)e.GetPosition(this).Y);
        }

        private void OnPointerMoved(object? sender, PointerEventArgs e)
        {
            if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
            {
                var currentPos = new Vector2D<float>((float)e.GetPosition(this).X, (float)e.GetPosition(this).Y);
                var delta = currentPos - _lastMousePos;

                _rotationY += delta.X * 0.005f;
                _rotationX += delta.Y * 0.005f;

                _lastMousePos = currentPos;
                RequestNextFrameRendering();
            }
        }

        private void OnPointerWheelChanged(object? sender, PointerWheelEventArgs e)
        {
            _zoom -= (float)e.Delta.Y * 0.5f;
            _zoom = Math.Max(1.0f, Math.Min(10.0f, _zoom));
            RequestNextFrameRendering();
        }

        public override void Render(DrawingContext context)
        {
            Render();
        }
    }
}
