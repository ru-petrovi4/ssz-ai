using Avalonia;
using Microsoft.Extensions.Hosting;
using System;
using Ssz.Utils.ConfigurationCrypter.Extensions;
using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using Microsoft.AspNetCore.Hosting;
using Ssz.AI.Grafana;
using Avalonia.Rendering.Composition.Animations;
using System.Threading.Tasks;
using System.IO;
using Ssz.Utils;
using Microsoft.Extensions.DependencyInjection;

namespace Ssz.AI
{
    internal sealed class Program
    {
        #region public functions 

        public static IHost Host { get; private set; } = null!;

        public static string EnvironmentName { get; private set; } = null!;

        #endregion

        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        [STAThread]
        public static void Main(string[] args)
        {
            Host = CreateHostBuilder(args).Build();

            var logger = Host.Services.GetRequiredService<ILogger<Program>>();
            logger.LogInformation($"App starting with args: \"{String.Join(" ", args)}\"; Environment: {EnvironmentName}; Working Directory: \"{Directory.GetCurrentDirectory()}\"");

            var t = Host.RunAsync();

            BuildAvaloniaApp()
                .StartWithClassicDesktopLifetime(args);
        }

        private static IHostBuilder CreateHostBuilder(string[] args)
        {
            return Microsoft.Extensions.Hosting.Host.CreateDefaultBuilder(args)
                .ConfigureAppConfiguration((hostingContext, config) =>
                {
                    EnvironmentName = ConfigurationHelper.GetEnvironmentName(hostingContext.HostingEnvironment);

                    config.Sources.Clear();

                    config.AddEncryptedAppSettings(hostingContext.HostingEnvironment, crypter =>
                    {
                        crypter.CertificatePath = "appsettings.pfx";                        
                    });
                })
                .ConfigureLogging(
                    builder =>
                        builder.ClearProviders()
                           .AddSszLogger()
                           .AddDebug()
                    )                
                .ConfigureWebHostDefaults(
                    webBuilder =>
                    {
                        webBuilder.UseStartup<Startup>();
                    });
        }

        // Avalonia configuration, don't remove; also used by visual designer.
        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .WithInterFont();
    }
}
