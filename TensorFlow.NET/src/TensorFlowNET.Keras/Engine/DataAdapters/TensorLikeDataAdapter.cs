﻿using System;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine.DataAdapters
{
    /// <summary>
    /// Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy.
    /// </summary>
    public class TensorLikeDataAdapter : DataAdapter, IDataAdapter
    {
        int _size;
        int _batch_size;
        int num_samples;
        int num_full_batches;
        int _partial_batch_size;

        public TensorLikeDataAdapter(DataAdapterArgs args)
        {
            this.args = args;
            Tensor sample_weight_tensor = args.SampleWeight != null ? _process_tensorlike(args.SampleWeight) : null;
            num_samples = (int)args.X.shape[0];
            var batch_size = args.BatchSize == -1 ? 32 : args.BatchSize;
            _batch_size = batch_size;
            _size = Convert.ToInt32(Math.Ceiling(num_samples / (batch_size + 0.0f)));
            num_full_batches = num_samples / batch_size;
            _partial_batch_size = num_samples % batch_size;

            var indices_dataset = tf.data.Dataset.range(1);
            indices_dataset = indices_dataset.repeat(args.Epochs);
            indices_dataset = indices_dataset.map(permutation).prefetch(1);
            indices_dataset = indices_dataset.flat_map(slice_batch_indices);
            var inputs = new Tensors();
            if (args.X != null)
                inputs.AddRange(args.X);
            if (args.Y != null)
                inputs.AddRange(args.Y);
            if (sample_weight_tensor != null)
                inputs.Add(sample_weight_tensor);
            dataset = slice_inputs(indices_dataset, inputs);
            dataset.FirstInputTensorCount = args.X.Length;
        }

        Tensors permutation(Tensors tensor)
        {
            var indices = math_ops.range(num_samples, dtype: dtypes.int64);
            if (args.Shuffle)
                indices = random_ops.random_shuffle(indices);
            return indices;
        }

        /// <summary>
        /// Convert a Tensor of indices into a dataset of batched indices.
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        IDatasetV2 slice_batch_indices(Tensor indices)
        {
            var num_in_full_batch = num_full_batches * _batch_size;
            var first_k_indices = array_ops.slice(indices, new Tensor[] { ops.convert_to_tensor(0) }, 
                new Tensor[] { ops.convert_to_tensor(num_in_full_batch) });
            first_k_indices = array_ops.reshape(first_k_indices, new int[] { num_full_batches, _batch_size });
            var flat_dataset = tf.data.Dataset.from_tensor_slices(first_k_indices);
            if (_partial_batch_size > 0)
            {
                var array = array_ops.slice(indices, 
                    new[] { constant_op.constant(num_in_full_batch)}, 
                    new[] { constant_op.constant(_partial_batch_size)});
                var index_remainder = tf.data.Dataset.from_tensors(array);
                flat_dataset = flat_dataset.concatenate(index_remainder);
            }
                
            return flat_dataset;
        }

        IDatasetV2 slice_inputs(IDatasetV2 indices_dataset, Tensors elements)
        {
            var dataset = tf.data.Dataset.from_tensors(elements).repeat();
            dataset = tf.data.Dataset.zip(indices_dataset, dataset);

            dataset = dataset.map(inputs =>
            {
                var indices = inputs[0];
                var results = inputs.Skip(1)
                    .Select(x => array_ops.gather(x, indices, axis: 0))
                    .ToArray();
                return new Tensors(results);
            }, -1);

            return dataset.with_options(new DatasetOptions { });
        }

        public override int GetSize() => _size;

        public override bool ShouldRecreateIterator() => false;

        Tensor _process_tensorlike(NDArray sample_weights)
        {
            return tf.convert_to_tensor(sample_weights);
        }
    }
}
