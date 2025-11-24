using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using Microsoft.Extensions.Configuration;
using Microsoft.AspNetCore.Authorization;
using System.IO;
using Microsoft.AspNetCore.Mvc;
using Ssz.AI.Grafana;

namespace Ssz.AI
{
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Architecture", "DV2002:Unmapped types", Justification = "System Class")]
    public class Startup
    {
        #region construction and destruction

        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        #endregion

        #region public functions

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddSingleton<DataToDisplayHolder>();
            //services.AddTransient<Ssz.AI.Models.AdvancedEmbeddingModel2.Model01>();

            IMvcCoreBuilder mvcBuilder = services.AddMvcCore(options =>
                {
                    options.UseCentralRoutePrefix(new RouteAttribute(RouteNamespace));
                })
                .AddNewtonsoftJson();
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env, IServiceProvider serviceProvider, IConfiguration configuration)
        {   
            app.UseRouting();
            
            app.UseEndpoints(endpoints =>
                {
                    endpoints.MapControllers(); //.RequireAuthorization(@"MainPolicy");                    
                });
        }

        #endregion        

        private const string RouteNamespace = @"/api/v3";
    }
}