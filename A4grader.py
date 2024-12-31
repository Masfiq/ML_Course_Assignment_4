run_my_solution = False
assignmentNumber = '4'

import os
import copy
import signal
import os
import numpy as np
import subprocess

if run_my_solution:
    import neuralnetworksA4 as nn
    # from A4mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')


    if False:  # Because only grading code in neuralnetworkA3.py.  Using jupyter nbconvert to run notebook

        import subprocess, glob, pathlib

        nb_name = f'*A{assignmentNumber}solution*.ipynb'
        # nb_name = '*.ipynb'
        filename = next(glob.iglob(nb_name), None)
        print(nb_name, filename)

        print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
        if not filename:
            raise Exception(f'Please rename your notebook file to A{assignmentNumber}solution.ipynb'.format(assignmentNumber))
        with open('notebookcode.py', 'w') as outputFile:
            comm = f'jupyter nbconvert --to script {nb_name} --stdout'
            # subprocess.call(shlex.split(comm), stdout=outputFile, shell=True)
            subprocess.call(comm.split(), stdout=outputFile, shell=True)
        # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
        import sys
        import ast
        import types
        with open('notebookcode.py') as fp:
            tree = ast.parse(fp.read(), 'eval')
        print('Removing all statements that are not function or class defs or import statements.')
        for node in tree.body[:]:
            if (not isinstance(node, ast.FunctionDef) and
                not isinstance(node, ast.Import) and
                not isinstance(node, ast.ImportFrom) and
                not isinstance(node, ast.ClassDef)):
                tree.body.remove(node)
        # Now write remaining code to py file and import it
        module = types.ModuleType('notebookcodeStripped')
        code = compile(tree, 'notebookcodeStripped.py', 'exec')
        sys.modules['notebookcodeStripped'] = module
        exec(code, module.__dict__)
        # import notebookcodeStripped as useThisCode
        from notebookcodeStripped import *

print('\n============================\n import neuralnetworksA4 as nn \n============================')
import neuralnetworksA4 as nn
    
try:
    
    if nn.NeuralNetwork and nn.NeuralNetworkClassifier:
        print('neuralnetworksA4.py defines NeuralNetwork and NeuralNetworkClassifier')

except:
    raise Exception('NeuralNetwork and NeuralNetworkClassifier classes are not defined in neuralnetworksA4.py')

# required_funcs = ['nn.NeuralNetwork', 'nn.NeuralNetworkClassifier']

# for func in required_funcs:
#     if func not in dir() or not callable(globals()[func]):
#         print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
#         print('  Check the spelling and capitalization of the function name.')


def test(points, runthis, correct_str, incorrect_str):
    if eval(runthis):
        print()
        print('-'*70)
        print(f'----  {points}/{points} points. {correct_str}')
        print('-'*70)
        return points
    else:
        print()
        print('-'*70)
        print(f'----  0/{points} points. {incorrect_str}')
        print('-'*70)
        return 0

for func in ['NeuralNetwork']:
    if func not in dir(nn):  #  or not callable(globals()[func]):
        print('CRITICAL ERROR: Class named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')
        break
    for method in ['_forward', 'get_performance_trace', '_gradient_f',
                    '_error_f', '_make_weights_and_views', 'train', 'use']:
        if method not in dir(nn.NeuralNetwork):
            print('CRITICAL ERROR: NeuralNetwork Function named \'{}\' is not defined'.format(method))
            print('  Check the spelling and capitalization of the function name.')
            
exec_grade = 0


#### constructor ##################################################################

runthis = '''
# Checking that NeuralNetworkClassifier is subcless of NeuralNetwork
'''

testthis = 'issubclass(nn.NeuralNetworkClassifier, nn.NeuralNetwork)'

pts = 10

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Correct class inheritance.',
                       'Incorrect class inheritance.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Call to issubclass caused exception\n')
    print(ex)

######################################################################    

runthis = '''
# Checking if the _forward function in NeuralNetworkClassifier is inherited from NeuralNetwork

import inspect
forward_func = [f for f in inspect.classify_class_attrs(nn.NeuralNetworkClassifier) if (f.name == 'forward' or f.name == '_forward')]
'''

testthis = 'forward_func[0].defining_class == nn.NeuralNetwork'

pts = 5

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'NeuralNetworkClassifier _forward function correctly inherited from NeuralNetwork.',
                       'NeuralNetworkClassifier _forward function should be inherited from NeuralNetwork.')

except Exception as ex:
    print(f'\n--- 0/{pts} points. Test raised the exception:\n')
    print(ex)


######################################################################

runthis = '''
# Checking if __str__ is overridden in NeuralNetworkClassifier
import inspect
str_func = [f for f in inspect.classify_class_attrs(nn.NeuralNetworkClassifier) if (f.name == '__str__')]
'''

testthis = 'str_func[0].defining_class == nn.NeuralNetworkClassifier'

pts = 5
      
try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'NeuralNetworkClassifier __str__ function correctly overridden in NeuralNetworkClassifier.',
                      'NeuralNetworkClassifier __str__ function should be overridden in NeuralNetworkClassifier.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Test raised the exception:\n')
    print(ex)


######################################################################

runthis = '''
# Checking if _gradient_f in NeuralNetworkClassifier is defined (overridden) in NeuralNetworkClassifier
import inspect
str_func = [f for f in inspect.classify_class_attrs(nn.NeuralNetworkClassifier) if (f.name == '_gradient_f')]
'''
testthis = 'str_func[0].defining_class == nn.NeuralNetworkClassifier'



pts = 5

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'NeuralNetworkClassifier _gradient_f function correctly defined in NeuralNetworkClassifier.',
                       'NeuralNetworkClassifier _gradient_f function should be defined in NeuralNetworkClassifier.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Test raised the exception:\n')
    print(ex)


######################################################################

runthis = '''
# Checking if _backpropagate in NeuralNetworkClassifier is inherited from NeuralNetwork
import inspect
str_func = [f for f in inspect.classify_class_attrs(nn.NeuralNetworkClassifier) if (f.name == '_backpropagate')]
'''
testthis = 'str_func[0].defining_class == nn.NeuralNetwork'

pts = 5      
try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'NeuralNetworkClassifier _backpropagate function correctly inherited from NeuralNetwork.',
                       'NeuralNetworkClassifier _backpropagate function should be inherited from  NeuralNetwork.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Test raised the exception:\n')
    print(ex)

######################################################################

runthis = '''
nnet = nn.NeuralNetworkClassifier(2, [], 5)
W_shapes = [W.shape for W in nnet.Ws]
correct = [(3, 5)]
'''
testthis = 'correct == W_shapes'      

pts = 10

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       f'W_shapes is correct value of {W_shapes}.',
                       f'W_shapes is incorrect values {W_shapes}.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetworkClassifier raised the exception\n')
    print(ex)
    
######################################################################

runthis = '''
nnet = nn.NeuralNetworkClassifier(2, [], 5)
G_shapes = [G.shape for G in nnet.Grads]
correct = [(3, 5)]
'''
testthis = 'correct == G_shapes'

pts = 10
try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       f'G_shapes is correct value of {G_shapes}',
                       f'G_shapes is incorrect values {G_shapes}.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Accessing nnet.Gs raised the exception\n')
    print(ex)


#### train  ##################################################################

runthis = '''
np.random.seed(42)
X = np.random.uniform(0, 1, size=(100, 2))
T = (np.abs(X[:, 0:1] - 0.5) > 0.3).astype(int)
nnet = nn.NeuralNetworkClassifier(2, [10, 5], len(np.unique(T)))
nnet.train(X, T, X, T, 20, method='scg')
last_error = nnet.get_performance_trace()[-1]
correct = 0.9297448356260026
'''
testthis = 'np.allclose(last_error, correct, atol=0.1)'

pts = 10
try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Correct values in performance_trace.',
                       f'Incorrect values of {last_error[0]} in performance_trace.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or get_performance_trace() raised the exception\n')
    print(ex)
    

######################################################################

runthis = '''
np.random.seed(43)
X = np.random.uniform(0, 1, size=(20, 2))
T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)
T[T == 0] = 10
T[T == 1] = 20
# Unique class labels are now 10 and 20!
nnet = nn.NeuralNetworkClassifier(2, [10, 5], 2)
nnet.train(X, T, X, T, 200, method='scg')
classes, probs = nnet.use(X)
correct_classes = \
np.array([[20], [20], [10], [20], [10], [20], [20], [10], [20], [10],
[20], [10], [20], [10], [20], [10], [20], [20], [20], [20]])
'''
pts = 10

testthis = 'np.allclose(classes, correct_classes, atol=0.1)'

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Correct values in classes.',
                       f'Incorrect values in classes.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or use raised the exception\n')
    print(ex)


runthis = '''
correct_probs = \
np.array([[5.02457605e-09, 9.99999995e-01], [8.62009522e-10, 9.99999999e-01],
[9.99999999e-01, 9.29113040e-10], [1.26059447e-09, 9.99999999e-01],
[1.00000000e+00, 6.39547085e-13], [2.36254327e-09, 9.99999998e-01],
[1.34693543e-09, 9.99999999e-01], [9.99999960e-01, 3.96256246e-08],
[7.25882159e-09, 9.99999993e-01], [1.00000000e+00, 1.41558454e-15],
[2.09966039e-10, 1.00000000e+00], [1.00000000e+00, 3.09418630e-16],
[2.01456195e-10, 1.00000000e+00], [1.00000000e+00, 2.09626683e-16],
[1.72899120e-09, 9.99999998e-01], [9.99999998e-01, 2.33382708e-09],
[2.19039065e-10, 1.00000000e+00], [2.38235718e-10, 1.00000000e+00],
[3.10731426e-09, 9.99999997e-01], [8.26588031e-10, 9.99999999e-01]])
'''
testthis = 'np.allclose(probs, correct_probs, atol=0.1)'

try:
    print('\n')
    print('='*80, f'\nTesting this for {pts} points:')
    print(runthis)
    exec(runthis)
    print(f'\n#  and test result with    {testthis}')
    exec_grade += test(pts, testthis,
                       'Correct values in probs.',
                       f'Incorrect values in probs.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or use raised the exception\n')
    print(ex)
    
name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 80')
print('='*70)

print('''

-- / 5 points. Experiment with the three different optimization methods,
               at least three hidden layer structures including [], two
               learning rates, and two numbers of epochs. Use verbose=False
               as an argument to train(). For scg, ignore the learning rate
               loop. Print a single line for each run showing method, number
               of epochs, learning rate, hidden layer structure, and percent
               correct for training, validation, and testing data.

__ / 5 points. Function make_mnist_classifier defined and used correctly.

__ / 5 points. Discuss your results. In your discussion, include observations about
                which method achieves the best result,
                which method seems to do best with fewer epochs,
                what common classification mistakes are made as shown in your confusion matrices, and
                do larger networks (more layers, more units) work better than small networks?

__ / 5 points. Train a network with values for method, learning rate, number of epochs,
               and a hidden layer structure with no more than 100 units in the first layer
               that you found work well. Extract the weight matrix from the first layer. Now,
               for each unit (column in the weight matrix) ignore the first row of bias weights
               and reshape the remaining weights into a 28 x 28 image for each unit and display
               them. Complete the function to draw the weight matrix for one unit using draw_digit
               as a guide, then use it in a loop to draw the weight matrices for each unit in
               the first layer of your network.
               Discuss what you see. Describe some of the images as patterns that could be
               useful for classifying particular digits.''')

print()
print('='*70)
print(f'{name} Results and Discussion Grade is ___ / 20')
print('='*70)


print()
print('='*70)
print(f'{name} FINAL GRADE is  _  / 100')
print('='*70)


print('''
Extra Credit (2 points possible): 

Extra Credit for 1 point:

Repeat the above experiments with a different classification data set.  Randonly partition
your data into training, validaton and test parts if not already provided.  Write in 
markdown cells descriptions of the data and your results.
of the data and your results.

Extra Credit for 1 point:

Train a network with values for method, learning rate, number of epochs, and a
hidden layer structure with no more than 100 units in the first layer that you
found work well.  Extract the weight matrix from the first layer.

Now, for each unit (column in the weight matrix) ignore the first row of bias weights and
reshape the remaining weights into a 28 x 28 image for each unit and display them.
Complete the following function to draw the weight matrix for one unit using `draw_digit`
as a guide, then use it in a loop to draw the weight matrices for each unit in the first
layer of your network.

Discuss what you see.  Describe some of the images as patterns that could be useful for classifying particular digits.
''')

print(f'\n{name} EXTRA CREDIT is 0 / 2')


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')


