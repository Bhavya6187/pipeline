function ret = softmax(in)
in = squeeze(in);
in = in - max(in);
in = exp(in);
div = sum(in);
ret = in/div;