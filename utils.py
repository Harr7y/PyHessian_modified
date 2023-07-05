import torch
import numpy as np
import torch.nn as nn


def normalization(v):
    """
    normalization of a list of vectors
    return: a list of normalized vectors v
    """
    v = [x / ((torch.sum(x * y)**0.5).item() + 1e-6) for (x, y) in zip(v, v)]
    return v

# v = [torch.randn((3,3,3,3))]
# sum_ = normalization(v)
# for s in sum_:
#     print(torch.norm(s))


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'weight' in name:
            params.append(param)
            grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def group_product_sum(xs, ys):
    """
    the inner product of two lists of vectors xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return: a list of the inner product of two lists of variables xs,ys
    """
    return [torch.sum(x * y) for (x, y) in zip(xs, ys)]



def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = []
    for i in range(len(gradsH)):
        hv.append(torch.autograd.grad(gradsH[i],
                                 params[i],
                                 grad_outputs=v[i],
                                 only_inputs=True,
                                 retain_graph=True)[0])
    return tuple(hv)


def dataloader_hv_product(v, model, dataloader, criterion):
    num_data = 0  # count the number of datum points in the dataloader
    params, gradsH = get_params_grad(model)
    THv = [torch.zeros(p.size()) for p in params]  # accumulate result
    for inputs, targets in dataloader:
        model.zero_grad()
        tmp_num_data = inputs.size(0)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward(create_graph=True)
        params, gradsH = get_params_grad(model)
        model.zero_grad()
        Hv = hessian_vector_product(gradsH, params, v)
        THv = [
            THv1 + Hv1 * float(tmp_num_data) + 0.
            for THv1, Hv1 in zip(THv, Hv)
        ]
        num_data += float(tmp_num_data)

    THv = [THv1 / float(num_data) for THv1 in THv]
    return THv

# Calculate the eigenvalues with the given inputs
def eigenvalue(model, input, target, criterion, tol=1e-4, iteration=100, topk=1):
    output = model(input)
    model.zero_grad()
    loss = criterion(output, target)
    loss.backward(create_graph=True, retain_graph=True)

    params, gradsH = get_params_grad(model)

    computed_dim, top_k = 0, topk
    eigenvalues = []
    eigenvectors = []

    while computed_dim < top_k:
        eigenvalue = None
        v = [torch.randn(p.size()) for p in params]
        v = normalization(v)
        for i in range(iteration):
            # only compute top-1 eigenvalue, I didn't consider the orthonormal now
            #         v = orthnormal(v, eigenvectors)
            model.zero_grad()
            Hv = hessian_vector_product(gradsH, params, v)
            tmp_eigenvalue = group_product(Hv, v)

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(sum(eigenvalue) - sum(tmp_eigenvalue)) / (abs(sum(eigenvalue) + 1e-6)) < tol:
                    print("Break: ", i)
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        computed_dim += 1

    return eigenvalues, eigenvectors

# Calculate the eigenvalues of the entire dataset
def eigenvalue_dataloader(model, dataloader, criterion, tol=1e-4, iteration=100, topk=1):
    params, gradsH = get_params_grad(model)
    computed_dim, top_k = 0, topk
    eigenvalues = []
    eigenvectors = []

    while computed_dim < top_k:
        eigenvalue = None
        v = [torch.randn(p.size()) for p in params]
        v = normalization(v)
        for i in range(iteration):
            # only compute top-1 eigenvalue, I didn't consider the orthonormal now
            #         v = orthnormal(v, eigenvectors)
            model.zero_grad()
            Hv = dataloader_hv_product(v, model, dataloader, criterion)
            tmp_eigenvalue = group_product(Hv, v)

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(sum(eigenvalue) - sum(tmp_eigenvalue)) / (abs(sum(eigenvalue) + 1e-6)) < tol:
                    print("Break: ", i)
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        computed_dim += 1

    return eigenvalues, eigenvectors

# below is a simplified version: operate on vectors(tensors), not on a list of vectors.
def normalization_v(v):
    """
    normalization of a vector
    return: a normalized vectors v
    """
    v = v / torch.norm(v)
    return v


def eliminate_projection_v(vector1, vector2):
    """
    vector1 = vector1 + vector2 * alpha.
    alpha is the ratio of the projection length of v1 in the direction of v2 to the norm(B)
    alpha = <v1, v2> / norm(v2)**2
    :return: v1 - v2 * alpha
    The component of v1 perpendicular to v2
    """
    #     alpha = - (torch.sum(x * y) for (x, y) in zip(params, update))
    alpha = - inner_product_v(vector1, vector2) / (torch.norm(vector2) ** 2)
    vector1.data.add_(vector2 * alpha)

    return vector1


def orthnormal_v(v, v_list):
    """
    make vector v orthogonal to each vector in v_list.
    Then, normalize the output v
    """
    for vv in v_list:
        v = eliminate_projection_v(v, vv)

    return normalization_v(v)


def inner_product_v(v1, v2):
    """
    :return: the inner product of two vectors v1, v2
    """
    return torch.sum(v1 * v2)


def hessian_vector_product_v(gradH, param, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    # return \frac{\partial gradH}{\partial param} \cdot v
    """
    return torch.autograd.grad(gradH, param, grad_outputs=v, only_inputs=True, retain_graph=True)[0]


# Calculate the eigenvalues with the given inputs
def eigen(model, input, target, criterion, iteration=100, topk=1, tol=1e-4):
    output = model(input)
    model.zero_grad()
    loss = criterion(output, target)
    loss.backward(create_graph=True, retain_graph=True)

    params, gradHs = get_params_grad(model)
    eigenvalues = [[] for _ in range(len(params))]
    eigenvectors = [[] for _ in range(len(params))]

    for i, (param, gradH) in enumerate(zip(params, gradHs)):
        computed_dim = 0
        while computed_dim < topk:
            eigenvalue = None
            v = torch.randn(param.size())
            v = normalization_v(v)
            for j in range(iteration):
                # 只考虑top-1 eigenvalue时, orthonormal可以先不考虑
                v = orthnormal_v(v, eigenvectors[i])
                model.zero_grad()
                Hv = hessian_vector_product_v(gradH, param, v)
                tmp_eigenvalue = inner_product_v(Hv, v)

                v = normalization_v(Hv)

                # eigenvalue =tmp_eigenvalue
                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                        print("Break: ", j)
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues[i].append(eigenvalue)
            eigenvectors[i].append(v)

            computed_dim += 1
        print("Computation of Top-1 eigenvalue of ", i, "-th parameter is doned.")

    return eigenvalues, eigenvectors


# Calculate the trace with the given inputs
def trace(model, input, target, criterion, iteration=100, tol=1e-4):
    output = model(input)
    model.zero_grad()
    loss = criterion(output, target)
    loss.backward(create_graph=True, retain_graph=True)

    params, gradHs = get_params_grad(model)
    trace_vhv = [[] for _ in range(len(params))]

    for i, (param, gradH) in enumerate(zip(params, gradHs)):
        trace = None
        for j in range(iteration):
            model.zero_grad()
            # generate Rademacher random variables
            v = torch.randint_like(param, high=2, device=param.device)
            for v_i in v:
                v_i[v_i == 0] = -1

            Hv = hessian_vector_product_v(gradH, param, v)
            trace_vhv[i].append(inner_product_v(Hv, v))

            if trace and abs(np.mean(trace_vhv[i]) - trace) / (abs(trace) + 1e-6) < tol:
                break
            else:
                trace = np.mean(trace_vhv[i])
        print("Computation of Trace of ", i, "-th parameter is doned.")
    traces = []
    for trace in trace_vhv:
        traces.append(np.mean(trace))

    return traces


# traces = trace(model, input, target, criterion)


# compute the accuracy of top-k
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res