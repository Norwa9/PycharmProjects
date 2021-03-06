import paddle
import paddle.fluid as fluid
import numpy

paddle.enable_static()

train_program = fluid.Program()
startup_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    data = fluid.data(name='X', shape=[None, 1], dtype='float32') # 数据层
    hidden = fluid.layers.fc(input=data, size=10) # 隐藏层（线性回归）
    loss = fluid.layers.mean(hidden) # 损失函数
    sgd = fluid.optimizer.SGD(learning_rate=0.001) #随机梯度下降
    sgd.minimize(loss)

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# Run the startup program once and only once.
# Not need to optimize/compile the startup program.
startup_program.random_seed=1
exe.run(startup_program)

# Run the main program directly without compile.
x = numpy.random.random(size=(10, 1)).astype('float32')
loss_data, = exe.run(train_program,
                     feed={"X": x},
                     fetch_list=[loss.name])

# Or use CompiledProgram:
# compiled_prog = fluid.CompiledProgram(train_program)
# loss_data, = exe.run(compiled_prog,
#              feed={"X": x},
#              fetch_list=[loss.name])

print(loss_data)