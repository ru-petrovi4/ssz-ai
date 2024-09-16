using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Threading.Tasks;

namespace Ssz.AI.Grafana
{
    [Route("grafana/json")]
    public partial class GrafanaJsonController : ControllerBase
    {
        #region construction and destruction

        public GrafanaJsonController(                        
            ILogger<GrafanaJsonController> logger)
        {
            _logger = logger;
        }

        #endregion

        //        #region public functions   

        public const string BatchesCount_PropertyName = @"BatchesCount";

        public const string BooleanHisogram_Metric = @"BooleanHisogram";
        public const string N_PropertyName = @"N";
        public const string M_PropertyName = @"M";

        public const string Distribution_Metric = @"Distribution";
        public const string Min_PropertyName = @"Min";

        public const string Cortex_Metric = @"Cortex";
        public const string CortexDisplayType_PropertyName = @"CortexDisplayType";

        public const string WordVectorOld_Metric = @"WordVectorOld";
        public const string WordVectorNew_Metric = @"WordVectorNew";
        public const string WordId_PropertyName = @"WordId";
        public const string WordNum_PropertyName = @"WordNum";
        public const string PrimaryWordsSelectionMethod_PropertyName = @"PrimaryWordsSelectionMethod";
        public const string PrimaryWordsCount_PropertyName = @"PrimaryWordsCount";
        public const string PrimaryWords_FinalVector_BitsCount_PropertyName = @"PrimaryWords_FinalVector_BitsCount";
        public const string SecondaryWords_FinalVector_BitsCount_PropertyName = @"SecondaryWords_FinalVector_BitsCount";

        public const string WordsVectorsComparisonOld_Metric = @"WordsVectorsComparisonOld";
        public const string WordsVectorsComparisonNew_Metric = @"WordsVectorsComparisonNew";
        public const string WordId1_PropertyName = @"WordId1";
        public const string WordId2_PropertyName = @"WordId2";

        public const string ClusterWordsCount_Metric = @"ClusterWordsCount";

        public const string ProxWordsNew_Metric = @"ProxWordsNew";
        public const string DotProductLow_PropertyName = @"DotProductLow";
        public const string DotProductDelta_PropertyName = @"DotProductDelta";
        public const string DotProductVariant_PropertyName = @"DotProductVariant";

        public const string EmbeddingForText_Metric = @"EmbeddingForText";
        public const string Text_PropertyName = @"Text";

        /// <summary>
        ///     Used for "Test connection" on the datasource config page.
        /// </summary>
        /// <returns></returns>
        [HttpGet]
        public Task<IActionResult> DatasourceHealthAsync()
        {
            return Task.FromResult<IActionResult>(StatusCode(200));
        }

        //        /// <summary>
        //        ///     Returns available metrics.
        //        /// </summary>
        //        /// <param name="listMetricsRequest"></param>
        //        /// <returns></returns>
        //        [HttpPost(@"metrics")]
        //        public IActionResult ListMetrics([FromBody] ListMetricsRequest listMetricsRequest)
        //        {
        //            var result = new List<ListMetricsResponse>
        //            {
        //                new ListMetricsResponse
        //                {
        //                    Label = "Boolean Vector Info",
        //                    Value = BooleanHisogram_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "N",
        //                            Name = N_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "integer",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "M",
        //                            Name = M_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "integer",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        }
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {                    
        //                    Label = "Distribution",
        //                    Value = Distribution_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Batches Count",
        //                            Name = BatchesCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "integer",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                            //Options = new List<ListMetricsResponsePayloadsOptions>
        //                            //{
        //                            //    new ListMetricsResponsePayloadsOptions
        //                            //    {
        //                            //        Label = "1000",
        //                            //        Value = "1000"
        //                            //    },
        //                            //    new ListMetricsResponsePayloadsOptions
        //                            //    {
        //                            //        Label = "100",
        //                            //        Value = "100"
        //                            //    }
        //                            //}
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Min",
        //                            Name = Min_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "integer",
        //                            ReloadMetric = true,
        //                            Width = 40,                            
        //                        }                        
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "Cortex Visualisation",
        //                    Value = Cortex_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "CortexDisplayType",
        //                            Name = CortexDisplayType_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        }
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "\"Старый\" вектор",
        //                    Value = WordVectorOld_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word ID",
        //                            Name = WordId_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        }
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "\"Новый\" вектор",
        //                    Value = WordVectorNew_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word Num",
        //                            Name = WordNum_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word ID",
        //                            Name = WordId_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWordsSelectionMethod",
        //                            Name = PrimaryWordsSelectionMethod_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWordsCount",
        //                            Name = PrimaryWordsCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWords_FinalVector_BitsCount",
        //                            Name = PrimaryWords_FinalVector_BitsCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },                        
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "SecondaryWords_FinalVector_BitsCount",
        //                            Name = SecondaryWords_FinalVector_BitsCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        }
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "Сравнение \"Старых\" векторов",
        //                    Value = WordsVectorsComparisonOld_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word 1 ID",
        //                            Name = WordId1_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word 2 ID",
        //                            Name = WordId2_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },                        
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "Сравнение \"Новых\" векторов",
        //                    Value = WordsVectorsComparisonNew_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {                        
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word 1 ID",
        //                            Name = WordId1_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Word 2 ID",
        //                            Name = WordId2_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 40,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWordsSelectionMethod",
        //                            Name = PrimaryWordsSelectionMethod_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWordsCount",
        //                            Name = PrimaryWordsCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWords_FinalVector_BitsCount",
        //                            Name = PrimaryWords_FinalVector_BitsCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "SecondaryWords_FinalVector_BitsCount",
        //                            Name = SecondaryWords_FinalVector_BitsCount_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        }
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "Количество слов в кластерах",
        //                    Value = ClusterWordsCount_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {                        
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWordsSelectionMethod",
        //                            Name = PrimaryWordsSelectionMethod_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },                        
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "Гистограмма общих бит",
        //                    Value = ProxWordsNew_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "PrimaryWordsSelectionMethod",
        //                            Name = PrimaryWordsSelectionMethod_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "DotProductLow",
        //                            Name = DotProductLow_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "float",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "DotProductDelta",
        //                            Name = DotProductDelta_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "float",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "DotProductVariant",
        //                            Name = DotProductVariant_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },
        //                    }
        //                },
        //                new ListMetricsResponse
        //                {
        //                    Label = "Эмбеддинг для текста",
        //                    Value = EmbeddingForText_Metric,
        //                    Payloads = new List<ListMetricsResponsePayload>
        //                    {
        //                        new ListMetricsResponsePayload
        //                        {
        //                            Label = "Text",
        //                            Name = Text_PropertyName,
        //                            Type = ListMetricsResponsePayload.TypeEnum.Input,
        //                            Placeholder = "string",
        //                            ReloadMetric = true,
        //                            Width = 60,
        //                        },                        
        //                    }
        //                },
        //            };
        //            return Ok(result);            
        //        }

        //        /// <summary>
        //        ///     Returns a list of metric payload options.
        //        /// </summary>
        //        /// <param name="listMetricPayloadOptionsRequest"></param>
        //        /// <returns></returns>
        //        [HttpPost(@"metric-payload-options")]
        //        public IActionResult ListMetricPayloadOptions([FromBody] ListMetricPayloadOptionsRequest listMetricPayloadOptionsRequest)
        //        {            
        //            string exampleJson = """
        //[{ 
        //  "label": "CPUUtilization",
        //  "value": "CPUUtilization"
        //},{
        //  "label": "DiskReadIOPS",
        //  "value": "DiskReadIOPS"
        //},{
        //  "label": "memory_freeutilization",
        //  "value": "memory_freeutilization"
        //}]
        //""";
        //            //var example = exampleJson != null
        //            //    ? JsonConvert.DeserializeObject<List<ListMetricPayloadOptionsResponse>>(exampleJson)
        //            //    : default(List<ListMetricPayloadOptionsResponse>);

        //            //var result = new List<ListMetricPayloadOptionsResponse>
        //            //{
        //            //    new ListMetricPayloadOptionsResponse
        //            //    {
        //            //        Value = "Value1",
        //            //        Payloads = new List<ListMetricsResponsePayloads>
        //            //        {
        //            //            new ListMetricsResponsePayloads
        //            //            {
        //            //                Name = "aaa"
        //            //            }
        //            //        }
        //            //    }
        //            //};
        //            //return new ObjectResult(result);

        //            return new ObjectResult(default(List<ListMetricPayloadOptionsResponse>));
        //        }

        //        /// <summary>
        //        ///     Returns panel data or annotations.
        //        /// </summary>
        //        /// <param name="queryRequest"></param>
        //        /// <returns></returns>
        //        [HttpPost(@"query")]
        //        public async Task<IActionResult> QueryAsync([FromBody] QueryRequest queryRequest)
        //        {
        //            if (queryRequest.Targets is null)
        //                return NoContent();

        //            List<QueryResponse> result = new();

        //            foreach (var target in queryRequest.Targets)
        //            {
        //                switch (target.Target)
        //                {
        //                    case BooleanHisogram_Metric:
        //                        result.Add(await Query_BooleanHisogram(queryRequest, target));
        //                        break;
        //                    case Distribution_Metric:
        //                        result.Add(await Query_DistributionAsync(queryRequest, target));
        //                        break;
        //                    case Cortex_Metric:
        //                        result.Add(await Query_CortexAsync(queryRequest, target));
        //                        break;
        //                    case WordVectorOld_Metric:
        //                        result.Add(await Query_WordVectorOldAsync(queryRequest, target));
        //                        break;
        //                    case WordVectorNew_Metric:
        //                        result.Add(await Query_WordVectorNewAsync(queryRequest, target));
        //                        break;
        //                    case WordsVectorsComparisonOld_Metric:
        //                        result.Add(await Query_WordsVectorsComparisonOldAsync(queryRequest, target));
        //                        break;
        //                    case WordsVectorsComparisonNew_Metric:
        //                        result.Add(await Query_WordsVectorsComparisonNewAsync(queryRequest, target));
        //                        break;
        //                    case ClusterWordsCount_Metric:
        //                        result.Add(await Query_ClusterWordsCountAsync(queryRequest, target));
        //                        break;
        //                    case ProxWordsNew_Metric:
        //                        result.Add(await Query_ProxWordsNewAsync(queryRequest, target));
        //                        break;
        //                    case EmbeddingForText_Metric:
        //                        result.Add(await Query_EmbeddingForTextAsync(queryRequest, target));
        //                        break;
        //                }
        //            }

        //            return Ok(result);
        //        }

        //        /// <summary>
        //        ///    Returns data for Variable of type Query.
        //        /// </summary>
        //        /// <param name="variableRequest"></param>
        //        /// <returns></returns>
        //        [HttpPost(@"variable")]
        //        public IActionResult Variable([FromBody] VariableRequest variableRequest)
        //        {            
        //            string exampleJson = """
        //[
        //  {"__text":"Label 1", "__value":"Value1"},
        //  {"__text":"Label 2", "__value":"Value2"},
        //  {"__text":"Label 3", "__value":"Value3"}
        //]
        //""";        
        //            //var example = exampleJson != null
        //            //    ? JsonConvert.DeserializeObject<VariableResponse>(exampleJson)
        //            //    : default(VariableResponse);            
        //            return Ok(exampleJson);
        //        }

        //        /// <summary>
        //        ///     Returning tag keys for ad hoc filters.
        //        /// </summary>
        //        /// <returns></returns>
        //        [HttpPost(@"tag-keys")]
        //        public IActionResult TagKeys()
        //        {
        //            string exampleJson = """
        //[
        //    {"type":"string","text":"City"},
        //    {"type":"string","text":"Country"}
        //]
        //""";
        //            //var example = exampleJson != null
        //            //    ? JsonConvert.DeserializeObject<List<TagKeysResponse>>(exampleJson)
        //            //    : default(List<TagKeysResponse>);            
        //            return Ok(exampleJson);
        //        }

        //        /// <summary>
        //        ///     Returning tag values for ad hoc filters.
        //        /// </summary>
        //        /// <param name="TagValuesRequest"></param>
        //        /// <returns></returns>
        //        /// <exception cref="NotImplementedException"></exception>
        //        [HttpPost(@"tag-values")]
        //        public IActionResult TagValues([FromBody] TagValuesRequest tagValuesRequest)
        //        {
        //            return new ObjectResult(new CaseInsensitiveDictionary<string>());
        //        }

        //        #endregion

        //        #region private functions

        //        private Task<QueryResponse> Query_BooleanHisogram(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            int n = new Any(jsonElement[N_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);
        //            int m = new Any(jsonElement[M_PropertyName]?.ToString() ?? @"100").ValueAsInt32(false);

        //            List<object[]> rows = new List<object[]>();
        //            for (int i = 0; i <= n; i += 1)
        //            {
        //                rows.Add([i.ToString(), ModelHelper.GetProbability(i, n, m)]);
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = Distribution_Metric,
        //                    //Datapoints = datapoints,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"OnesCount", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"Probability", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return Task.FromResult(queryResponse);
        //        }

        //        private Task<QueryResponse> Query_DistributionAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            int batchesCount = new Any(jsonElement[BatchesCount_PropertyName]?.ToString() ?? @"1000").ValueAsInt32(false);
        //            float minDesired = new Any(jsonElement[Min_PropertyName]?.ToString() ?? @"-1.0").ValueAs<float>(false);

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;
        //            float min = -1.0f;
        //            float max = 1.0f;
        //            float delta = (max - min) / batchesCount;
        //            int[] resultInt = new int[batchesCount + 1];
        //            for (int index1 = 0; index1 < model.Words.Count; index1 += 1)
        //            {
        //                int indexBias = index1 * model.Words.Count;
        //                for (int index2 = 0; index2 < model.Words.Count; index2 += 1)
        //                {
        //                    if (index2 == index1)
        //                        continue;

        //                    int i = (int)((model.ProxWordsOldMatrix[indexBias + index2] - min) / delta);
        //                    resultInt[i] += 1;
        //                }
        //            }
        //            resultInt[batchesCount - 1] += resultInt[batchesCount];

        //            //var datapoints = new List<List<decimal>>(batchesCount);
        //            //for (int i = 0; i < batchesCount; i += 1)
        //            //{
        //            //    datapoints.Add(new List<decimal> { (decimal)(minAbs + delta * i), resultInt[i] });
        //            //}
        //            List<object[]> rows = new List<object[]>();
        //            for (int i = 0; i < batchesCount; i += 1)
        //            {
        //                var x = min + delta * i;
        //                if (x > minDesired)
        //                    rows.Add([x, resultInt[i]]);
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = Distribution_Metric,
        //                    //Datapoints = datapoints,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"Interval", Type = QueryResponseColumn.TypeEnum.Number },
        //                        new QueryResponseColumn { Text = @"Count", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };            

        //            return Task.FromResult(queryResponse);
        //        }

        //        /// <summary>
        //        ///     axis types: 'linear', 'log', 'category', 'date', 'autotyp'
        //        /// </summary>
        //        /// <param name="queryRequest"></param>
        //        /// <param name="queryRequestTarget"></param>
        //        /// <returns></returns>
        //        private Task<QueryResponse> Query_CortexAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            CortexDisplayType cortexDisplayType = new Any(jsonElement[CortexDisplayType_PropertyName]?.ToString()).ValueAs<CortexDisplayType>(false);

        //            List<object[]> rows = new List<object[]>();

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;
        //            lock (model.CortexCopySyncRoot)
        //            {
        //                switch (cortexDisplayType)
        //                {
        //                    case CortexDisplayType.GroupId_ToDisplay:
        //                        foreach (var point in model.CortexCopy.Array)
        //                        {
        //                            rows.Add([point.V[0], point.V[1], $"{ColorFromGroupId_ToDisplay(point.GroupId_ToDisplay)}"]);
        //                        }
        //                        break;
        //                    case CortexDisplayType.Spot:
        //                        Point? selectedPoint = model.CortexCopy.Array.FirstOrDefault(p => p.GroupId_ToDisplay == (int)PointGroupId_ToDisplay.MainPoint1);
        //                        if (selectedPoint != null && selectedPoint.WordIndex >= 0)
        //                        {
        //                            int bias = selectedPoint.WordIndex * model.Words.Count;
        //                            foreach (var point in model.CortexCopy.Array)
        //                            {
        //                                if (point.WordIndex != -1)
        //                                    rows.Add([point.V[0], point.V[1], $"{ColorFromDotProduct(model.ProxWordsOldMatrix[bias + point.WordIndex])}"]);
        //                            }
        //                        }
        //                        break;
        //                }                
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = Cortex_Metric,
        //                    //Datapoints = datapoints,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"X", Type = QueryResponseColumn.TypeEnum.Number },
        //                        new QueryResponseColumn { Text = @"Y", Type = QueryResponseColumn.TypeEnum.Number },
        //                        new QueryResponseColumn { Text = @"Color", Type = QueryResponseColumn.TypeEnum.String },
        //                    },
        //                    Rows = rows,
        //                };

        //            return Task.FromResult(queryResponse);
        //        }        

        //        private Task<QueryResponse> Query_WordVectorOldAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            int wordId = new Any(jsonElement[WordId_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);
        //            string wordName = jsonElement[WordId_PropertyName]?.ToString() ?? @"";

        //            List<object[]> rows = new List<object[]>();

        //            Word? word = null;
        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;
        //            if (wordId > 0 && wordId < model.Words.Count) 
        //            {
        //                word = model.Words[wordId];                
        //            }
        //            else
        //            {
        //                word = model.Words.FirstOrDefault(w => w.Name == wordName);
        //            }

        //            if (word is not null)
        //            {
        //                var oldVectror = word.OldVector;
        //                for (int i = 0; i < oldVectror.Length; i += 1)
        //                {
        //                    rows.Add([i.ToString(), oldVectror[i]]);
        //                }
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorOld_Metric,                    
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"i", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"vector[i]", Type = QueryResponseColumn.TypeEnum.Number },                        
        //                    },
        //                    Rows = rows,
        //                };

        //            return Task.FromResult(queryResponse);
        //        }

        //        private async Task<QueryResponse> Query_WordVectorNewAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            int wordId = new Any(jsonElement[WordId_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);
        //            string wordName = jsonElement[WordId_PropertyName]?.ToString() ?? @"";

        //            int wordNum = new Any(jsonElement[WordNum_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);
        //            Clusterization_AlgorithmEnum primaryWordsSelectionMethod = new Any(jsonElement[PrimaryWordsSelectionMethod_PropertyName]?.ToString() ?? "").ValueAs<Clusterization_AlgorithmEnum>(false);
        //            int primaryWordsCount = new Any(jsonElement[PrimaryWordsCount_PropertyName]?.ToString() ?? @"1000").ValueAsInt32(false);
        //            int primaryWords_FinalVector_BitsCount = new Any(jsonElement[PrimaryWords_FinalVector_BitsCount_PropertyName]?.ToString() ?? @"8").ValueAsInt32(false);            
        //            int secondaryWords_FinalVector_BitsCount = new Any(jsonElement[SecondaryWords_FinalVector_BitsCount_PropertyName]?.ToString() ?? @"8").ValueAsInt32(false);

        //            List<object[]> rows = new List<object[]>();

        //            Word? word = null;
        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;
        //            if (wordId > 0 && wordId < model.Words.Count)
        //            {
        //                word = model.Words[wordId];
        //            }
        //            else
        //            {
        //                word = model.Words.FirstOrDefault(w => w.Name == wordName);
        //            }

        //            if (word is not null)
        //            {
        //                var taskCompletionSource = new TaskCompletionSource();

        //                _jobsManager.MainBackgroundService_ThreadSafeDispatcher.BeginInvoke(ct =>
        //                {
        //                    model.PrimaryWordsSelectionMethod = primaryWordsSelectionMethod;

        //                    model.Calculate_NewVector_ToDisplay(word, wordNum);

        //                    taskCompletionSource.SetResult();
        //                });

        //                await taskCompletionSource.Task;

        //                if (word.NewVector_ToDisplay is not null)
        //                {
        //                    for (int i = 0; i < word.NewVector_ToDisplay.Length; i += 1)
        //                    {
        //                        rows.Add([i.ToString(), word.NewVector_ToDisplay[i]]);
        //                    }
        //                }
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorNew_Metric,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"i", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"vector[i]", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return queryResponse;
        //        }

        //        private async Task<QueryResponse> Query_WordsVectorsComparisonOldAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            int wordId1 = new Any(jsonElement[WordId1_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);
        //            string wordName1 = jsonElement[WordId1_PropertyName]?.ToString() ?? @"";
        //            int wordId2 = new Any(jsonElement[WordId2_PropertyName]?.ToString() ?? @"0").ValueAsInt32(false);
        //            string wordName2 = jsonElement[WordId2_PropertyName]?.ToString() ?? @"";

        //            List<object[]> rows = new List<object[]>();

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;

        //            Word? word1 = null;
        //            if (wordId1 > 0 && wordId1 < model.Words.Count)
        //            {
        //                word1 = model.Words[wordId1];
        //            }
        //            else
        //            {
        //                word1 = model.Words.FirstOrDefault(w => w.Name == wordName1);
        //            }
        //            Word? word2 = null;
        //            if (wordId2 > 0 && wordId2 < model.Words.Count)
        //            {
        //                word2 = model.Words[wordId2];
        //            }
        //            else
        //            {
        //                word2 = model.Words.FirstOrDefault(w => w.Name == wordName2);
        //            }

        //            if (word1 is not null && word2 is not null)
        //            {
        //                rows.Add(["DotProduct", model.ProxWordsOldMatrix[word1.Index * model.Words.Count + word2.Index]]);
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorNew_Metric,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"Name", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"Value", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return queryResponse;
        //        }

        //        private async Task<QueryResponse> Query_WordsVectorsComparisonNewAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;            
        //            string wordName1 = jsonElement[WordId1_PropertyName]?.ToString() ?? @"";            
        //            string wordName2 = jsonElement[WordId2_PropertyName]?.ToString() ?? @"";

        //            Clusterization_AlgorithmEnum primaryWordsSelectionMethod = new Any(jsonElement[PrimaryWordsSelectionMethod_PropertyName]?.ToString() ?? "").ValueAs<Clusterization_AlgorithmEnum>(false);
        //            int primaryWordsCount = new Any(jsonElement[PrimaryWordsCount_PropertyName]?.ToString() ?? @"1000").ValueAsInt32(false);
        //            int primaryWords_FinalVector_BitsCount = new Any(jsonElement[PrimaryWords_FinalVector_BitsCount_PropertyName]?.ToString() ?? @"8").ValueAsInt32(false);
        //            int secondaryWords_FinalVector_BitsCount = new Any(jsonElement[SecondaryWords_FinalVector_BitsCount_PropertyName]?.ToString() ?? @"8").ValueAsInt32(false);

        //            List<object[]> rows = new List<object[]>();

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;

        //            Word? word1 = model.Words.FirstOrDefault(w => w.Name == wordName1);            
        //            Word? word2 = model.Words.FirstOrDefault(w => w.Name == wordName2);

        //            if (word1 is not null && word2 is not null)
        //            {
        //                var taskCompletionSource = new TaskCompletionSource();

        //                _jobsManager.MainBackgroundService_ThreadSafeDispatcher.BeginInvoke(ct =>
        //                {
        //                    model.PrimaryWordsSelectionMethod = primaryWordsSelectionMethod;

        //                    model.Calculate_NewVector_ToDisplay(word1, 1);

        //                    model.Calculate_NewVector_ToDisplay(word2, 2);                    

        //                    taskCompletionSource.SetResult();
        //                });

        //                await taskCompletionSource.Task;

        //                if (word1.NewVector_ToDisplay is not null && word2.NewVector_ToDisplay is not null)
        //                {
        //                    int result = 0;
        //                    for (int i = 0; i < word1.NewVector_ToDisplay.Length; i += 1)
        //                    {
        //                        if (word1.NewVector_ToDisplay[i] > 0.0f && word2.NewVector_ToDisplay[i] > 0.0f)
        //                            result += 1;
        //                    }
        //                    rows.Add(["N", result]);
        //                }                
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorNew_Metric,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"Name", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"Value", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return queryResponse;
        //        }

        //        private async Task<QueryResponse> Query_ClusterWordsCountAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            Clusterization_AlgorithmEnum primaryWordsSelectionMethod = new Any(jsonElement[PrimaryWordsSelectionMethod_PropertyName]?.ToString() ?? "").ValueAs<Clusterization_AlgorithmEnum>(false);            

        //            List<object[]> rows = new List<object[]>();

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;

        //            model.PrimaryWordsSelectionMethod = primaryWordsSelectionMethod;            

        //            if (model.CurrentClusterization_Algorithm_ToDisplay?.PrimaryWords is not null)
        //            {
        //                WordCluster[] wordClusters = new WordCluster[model.PrimaryWordsCount];
        //                for (int clusterIndex = 0; clusterIndex < wordClusters.Length; clusterIndex += 1)
        //                {
        //                    WordCluster wordClustrer = new()
        //                    {
        //                        CentroidOldVector = new float[Model.OldVectorLength],
        //                    };
        //                    Array.Copy(model.CurrentClusterization_Algorithm_ToDisplay.PrimaryWords[clusterIndex].OldVector, wordClustrer.CentroidOldVector, Model.OldVectorLength);
        //                    wordClusters[clusterIndex] = wordClustrer;
        //                }

        //                int[] newClusterIndices = new int[model.Words.Count];

        //                Parallel.For(0, model.Words.Count, wordIndex =>
        //                {
        //                    Word word = model.Words[wordIndex];
        //                    var oldVectror = word.OldVector;
        //                    int nearestClusterIndex = -1;
        //                    float nearestDotProduct = 0.0f;
        //                    for (int clusterIndex = 0; clusterIndex < wordClusters.Length; clusterIndex += 1)
        //                    {
        //                        var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
        //                        float dotProduct = TensorPrimitives.Dot(oldVectror, wordCluster_CentroidOldVector);
        //                        if (dotProduct > nearestDotProduct)
        //                        {
        //                            nearestDotProduct = dotProduct;
        //                            nearestClusterIndex = clusterIndex;
        //                        }
        //                    }
        //                    wordClusters[nearestClusterIndex].WordsCount += 1;
        //                    newClusterIndices[wordIndex] = nearestClusterIndex;
        //                });

        //                await Task.Delay(0);

        //                for (int clusterIndex = 0; clusterIndex < wordClusters.Length; clusterIndex += 1)
        //                {
        //                    rows.Add([model.CurrentClusterization_Algorithm_ToDisplay.PrimaryWords[clusterIndex].Name, wordClusters[clusterIndex].WordsCount]);
        //                }
        //            }                       

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorNew_Metric,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"ClusterWord", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"WordsCount", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return queryResponse;
        //        }              

        //        private async Task<QueryResponse> Query_ProxWordsNewAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;
        //            Clusterization_AlgorithmEnum primaryWordsSelectionMethod = new Any(jsonElement[PrimaryWordsSelectionMethod_PropertyName]?.ToString() ?? "").ValueAs<Clusterization_AlgorithmEnum>(false);
        //            float dotProductLow = new Any(Regex.Unescape(jsonElement.Value<string>(DotProductLow_PropertyName) ?? @"")).ValueAsSingle(false);
        //            float dotProductDelta = new Any(Regex.Unescape(jsonElement.Value<string>(DotProductDelta_PropertyName) ?? @"")).ValueAsSingle(false);
        //            DotProductVariant dotProductVariant = new Any(Regex.Unescape(jsonElement.Value<string>(DotProductVariant_PropertyName) ?? @"")).ValueAs<DotProductVariant>(false);

        //            List<object[]> rows = new List<object[]>();

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;

        //            //model.PrimaryWordsSelectionMethod = primaryWordsSelectionMethod;

        //            var proxWordsOldMatrix = model.ProxWordsOldMatrix;
        //            float[]? proxWordsNewMatrix = null;
        //            switch (dotProductVariant)
        //            {
        //                case DotProductVariant.All:
        //                    proxWordsNewMatrix = model.CurrentNewVectorsAndMatrices_ToDisplay?.ProxWordsNewMatrix;
        //                    break;
        //                case DotProductVariant.PrimaryOnly:
        //                    proxWordsNewMatrix = model.CurrentNewVectorsAndMatrices_ToDisplay?.ProxWordsNewMatrix_PrimaryOnly;
        //                    break;
        //                case DotProductVariant.SecondaryOnly:
        //                    proxWordsNewMatrix = model.CurrentNewVectorsAndMatrices_ToDisplay?.ProxWordsNewMatrix_SecondaryOnly;
        //                    break;
        //            }            
        //            if (proxWordsOldMatrix is not null &&                
        //                proxWordsNewMatrix is not null)
        //            {
        //                int[] pairsCounts = new int[model.PrimaryWords_NewVector_BitsCount + model.SecondaryWords_NewVector_BitsCount + 1];

        //                float dotProductHigh = dotProductLow + dotProductDelta;
        //                var matrixArrayIndices = proxWordsOldMatrix
        //                    .Select((dp, i) => (dp, i))
        //                    .Where(it => it.Item1 >= dotProductLow && it.Item1 < dotProductHigh)                        
        //                    .ToArray();

        //                Parallel.For(0, pairsCounts.Length, commonBitsCount =>
        //                {
        //                    pairsCounts[commonBitsCount] = matrixArrayIndices
        //                        .Where(it => (int)Math.Round(proxWordsNewMatrix[it.Item2], 0) == commonBitsCount)
        //                        .Count();
        //                });

        //                await Task.Delay(0);

        //                for (int commonBitsCount = 0; commonBitsCount < pairsCounts.Length; commonBitsCount += 1)
        //                {
        //                    rows.Add([commonBitsCount.ToString(), pairsCounts[commonBitsCount]]);
        //                }
        //            }

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorNew_Metric,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"CommonBitsCount", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"PairsCount", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return queryResponse;
        //        }

        //        private async Task<QueryResponse> Query_EmbeddingForTextAsync(QueryRequest queryRequest, QueryRequestTarget queryRequestTarget)
        //        {
        //            var jsonElement = (JObject)queryRequestTarget.Payload!;            
        //            string text = Regex.Unescape(jsonElement.Value<string>(Text_PropertyName) ?? @"");

        //            List<object[]> rows = new List<object[]>();

        //            var model = _addonsManager.AddonsThreadSafe.OfType<ModelAddon>().Single().Model!;

        //            float[] newVector_ToDisplay = model.GetEmbeddingForPhrase(text);
        //            if (newVector_ToDisplay is not null)
        //            {
        //                for (int i = 0; i < newVector_ToDisplay.Length; i += 1)
        //                {
        //                    rows.Add([i.ToString(), newVector_ToDisplay[i]]);
        //                }
        //            }

        //            await Task.Delay(0);

        //            var queryResponse =
        //                new QueryResponse
        //                {
        //                    Target = WordVectorNew_Metric,
        //                    Type = QueryResponse.TypeEnum.Table,
        //                    Columns = new List<QueryResponseColumn>
        //                    {
        //                        new QueryResponseColumn { Text = @"i", Type = QueryResponseColumn.TypeEnum.String },
        //                        new QueryResponseColumn { Text = @"vector[i]", Type = QueryResponseColumn.TypeEnum.Number },
        //                    },
        //                    Rows = rows,
        //                };

        //            return queryResponse;
        //        }

        //        private string ColorFromDotProduct(float dotProduct)
        //        {
        //            string p = ((int)(dotProduct * 255)).ToString("X2");
        //            return "#" + p + p + p;
        //        }

        //        private string ColorFromGroupId_ToDisplay(int groupId_ToDisplay)
        //        {
        //            //switch (groupId) 
        //            //{
        //            //    case 1:
        //            //        return "#E05A31";
        //            //    case 2:
        //            //        return "#E0CE31";
        //            //    case 3:
        //            //        return "#B2E031";
        //            //    case 4:
        //            //        return "#31E05C";
        //            //    case 5:
        //            //        return "#31D7E0";
        //            //    case 6:
        //            //        return "#31AEE0";
        //            //    default:
        //            //        return "#220000";
        //            //}
        //            switch (groupId_ToDisplay)
        //            {
        //                case 1:
        //                    return "#FF1A00";
        //                case 2:
        //                    return "#E5FF00";
        //                case 3:
        //                    return "#00FF1A";
        //                case 4:
        //                    return "#003EFF";
        //                case 5:
        //                    return "#FFFFFF";

        //                case (int)PointGroupId_ToDisplay.PrimaryPoint:
        //                    return "#000000";

        //                case (int)PointGroupId_ToDisplay.MainPoint1:
        //                    return "#FFFFFF"; // White
        //                case (int)PointGroupId_ToDisplay.PrimaryPoint_Selected1:
        //                    return "#FF0000"; // Light Red
        //                case (int)PointGroupId_ToDisplay.SecondaryPoint_Selected1:
        //                    return "#8370D8"; // Light Light Blue
        //                case (int)PointGroupId_ToDisplay.PrimaryAndSecondaryPoint_Selected1:
        //                    return "#40EB34"; // Green

        //                //case (int)PointGroupId.MainPoint1:
        //                //    return "#A60000"; // Red
        //                //case (int)PointGroupId.PrimaryPoint_Selected1:
        //                //    return "#FF0000"; // Light Red
        //                //case (int)PointGroupId.SecondaryPoint_Selected1:
        //                //    return "#FF7373"; // Light Light Red
        //                //                      // 
        //                //case (int)PointGroupId.MainPoint2:
        //                //    return "#1B0773"; // Blue
        //                //case (int)PointGroupId.PrimaryPoint_Selected2:
        //                //    return "#3216B0"; // Light Blue
        //                //case (int)PointGroupId.SecondaryPoint_Selected2:
        //                //    return "#8370D8"; // Light Light Blue  

        //                default:
        //                    return "#220000";
        //            }
        //        }

        //        #endregion

        #region private fields

        private readonly ILogger _logger;        

        #endregion                
    }    
}




//HttpContext.Request.EnableBuffering();
//HttpContext.Request.Body.Position = 0;
//var rawRequestBody = await new StreamReader(HttpContext.Request.Body).ReadToEndAsync();


// metrics
//string exampleJson = """
//[{
//  "label": "Describe metric list", // Optional. If the value is empty, use the value as the label
//  "value": "DescribeMetricList", // The value of the option.
//  "payloads": [{ // Configuration parameters of the payload.
//    "label": "Namespace", // The label of the payload. If the value is empty, use the value as the label.
//    "name": "namespace", // The name of the payload. If the value is empty, use the name as the label.
//    "type": "select", // If the value is select, the UI of the payload is a radio box. If the value is multi-select, the UI of the payload is a multi selection box; if the value is input, the UI of the payload is an input box; if the value is textarea, the UI of the payload is a multiline input box. The default is input.
//    "placeholder": "Please select namespace", // Input box / selection box prompt information.
//    "reloadMetric": true, // Whether to overload the metrics API after modifying the value of the payload.
//    "width": 10, // Set the input / selection box width to a multiple of 8px. 
//    "options": [{ // If the payload type is select / multi-select, the list is the configuration of the option list.
//      "label": "acs_mongodb", // The label of the payload select option.
//      "value": "acs_mongodb", // The label of the payload value.
//        },{
//          "label": "acs_rds",
//          "value": "acs_rds",
//        }]
//      },{
//        "name": "metric",
//        "type": "select"
//      },{
//        "name": "instanceId",
//        "type": "select"
//      }]
//},{
//  "value": "DescribeMetricLast",
//  "payloads": [{
//    "name": "namespace",
//    "type": "select"
//  },{
//    "name": "metric",
//    "type": "select"
//  },{
//    "name": "instanceId",
//    "type": "multi-select"
//  }]
//}]
//""";
////var example = exampleJson != null
////    ? JsonConvert.DeserializeObject<List<ListMetricsResponse>>(exampleJson)
////    : default(List<ListMetricsResponse>);   
///

//[DataContract]
//public class PlotlyData
//{
//    [DataMember(Name = "x", EmitDefaultValue = true)]
//    public float[] X { get; set; } = null!;

//    [DataMember(Name = "y", EmitDefaultValue = true)]
//    public float[] Y { get; set; } = null!;

//    /// <summary>
//    /// Gets or Sets Type
//    /// </summary>
//    [TypeConverter(typeof(CustomEnumConverter<ModeEnum>))]
//    [JsonConverter(typeof(Newtonsoft.Json.Converters.StringEnumConverter))]
//    public enum ModeEnum
//    {
//        /// <summary>
//        /// Enum TableEnum for table
//        /// </summary>
//        [EnumMember(Value = "markers")] // Attribute is not supported in System.Text.Json
//        Markers = 1
//    }

//    /// <summary>
//    /// Gets or Sets Mode
//    /// </summary>
//    [Required]
//    [DataMember(Name = "mode", EmitDefaultValue = true)]
//    public ModeEnum Mode { get; set; }

//    /// <summary>
//    /// Gets or Sets Type
//    /// </summary>
//    [TypeConverter(typeof(CustomEnumConverter<TypeEnum>))]
//    [JsonConverter(typeof(Newtonsoft.Json.Converters.StringEnumConverter))]
//    public enum TypeEnum
//    {
//        /// <summary>
//        /// Enum TableEnum for table
//        /// </summary>
//        [EnumMember(Value = "scatter")] // Attribute is not supported in System.Text.Json
//        Scatter = 1
//    }

//    /// <summary>
//    /// Gets or Sets Type
//    /// </summary>
//    [Required]
//    [DataMember(Name = "type", EmitDefaultValue = true)]
//    public TypeEnum Type { get; set; }
//}