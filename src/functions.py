import torch
import torch.nn.functional as F

def conv2d(input, filter, stride=1, padding=0):
    
    # Assume the input shape is (batch_size, in_channels, in_height, in_width)
    # Assume the kernel shape is (out_channels, in_channels, kernel_height, kernel_width)
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, filter_height, filter_width = filter.shape

    # Calculate the output height and width while considering the effect of padding and 
    # stride on the dimensions of the output.
    output_height = (in_height + 2 * padding - filter_height) // stride + 1
    output_width = (in_width + 2 * padding - filter_width) // stride + 1

    # Add zero-padding around the edges of each input image in the batch, so that the 
    # filter can apply to the border pixels.
    input_padded = F.pad(input, (padding, padding, padding, padding))

    # Initialize the unfolded tensor for all patches
    unfolded_tensor = torch.zeros(
        (
            batch_size,
            in_channels * filter_height * filter_width,
            output_height * output_width,
        )
    )

    # Here we slide the filter over the input, and for each position we flatten the patch
    # of the image it covers and store it in the unfolded tensor.
    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + filter_height
            end_j = start_j + filter_width
            patches = input_padded[:, :, start_i:end_i, start_j:end_j]
            unfolded_tensor[:, :, i * output_width + j] = patches.reshape(
                batch_size, -1
            )

    # This reshapes the filter to be two-dimensional, with the first dimension being 
    # the number of output channels and the second being all the weights in the filter.
    filter_reshaped = (
        filter.view(out_channels, -1).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    
    # This is the core of the convolution operation: a matrix multiplication between the 
    # unfolded input patches and the reshaped filter.
    output = torch.bmm(filter_reshaped, unfolded_tensor)

    # Reshape the convolved output to the expected output shape
    output = output.view(batch_size, out_channels, output_height, output_width)

    return output


def maxpool2d(input, kernel_size=2, stride=2, padding=0):
    batch_size, in_channels, in_height, in_width = input.shape
    
    # Calculate the output height and width
    output_height = (in_height + 2 * padding - kernel_size) // stride + 1
    output_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # Pad the input
    input_padded = F.pad(input, (padding, padding, padding, padding))
    
    # Initialize the unfolded tensor for all patches
    unfolded_tensor = torch.zeros((batch_size, in_channels, kernel_size * kernel_size, output_height * output_width))
    
    # Extract patches
    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            patches = input_padded[:, :, start_i:end_i, start_j:end_j]
            unfolded_tensor[:, :, :, i * output_width + j] = patches.reshape(batch_size, in_channels, -1)
    
    # Perform max pooling
    pooled_output = unfolded_tensor.max(dim=2)[0]
    
    # Reshape the pooled output to the expected output shape
    pooled_output = pooled_output.view(batch_size, in_channels, output_height, output_width)
    
    return pooled_output

