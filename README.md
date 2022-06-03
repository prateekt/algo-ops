# Algo-Ops

Pipeline Infrastructure for Prototyping and Deploying Algorithms

You like algorithms. But prototyping complicated algorithms can be challenging, especially when they have complicated
architectures with many components. How do you prototype algorithms so that the data flow is sensible, the results are
interpretable, and the algorithm is easily debuggable on incorrect predictions?

Algo-Ops has the following features:

* An algorithm is a recipe for executing a computation, generally consisting of several steps. In Algo-Ops, each step is
  an Op.
* Ops are automic units of an algorithm. Various types of Ops are supported such as TextOps (for NLP) and CVOps (for
  computer vision). The user can easily add their own Ops.
* Ops are linked together to form an algorithm. They execute sequentially where the first Op's outputs are passed as the
  second Op's inputs. The feed-forward pipeline is run to execution.

```python
pipeline = Pipeline.init_from_funcs(
    funcs=[self.append_a, self.append_something,
           self.reverse, self.reverse],
    op_class=TextOp,
)
```

* Algo-Ops pipelines automatically dashboard in a Jupyter notebook. Simply do pipeline.vis() to visualize the data flow
  in a Jupyter notebook, making it easy to visualize data flow and debug an algorithmic edge case on some input.

```python
pipeline.vis()
```

* Algo-Ops pipelines automatically keep a profile of their own performance over time. Simply do pipeline.vis_profile()
  to see a profile of your Op executions in terms of runtime and memory usage.

```python
pipeline.vis_profile()
```

* Algo-Ops pipeline supports easy debugging. A pipeline can be run on a set of supervised inputs with known outputs. In
  this case, if the algorithm got the wrong prediction, the entire pipeline dataflow is auto-pickled to allow the user
  to investigate where the algorithm went wrong.
* An Algo-Ops pipeline is itself an Op, so pipelines themselves can be stacked to create very intricate workflows that
  are still easy to track and debug. The entire framework is highly extensible.
