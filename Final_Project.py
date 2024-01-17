from mpi4py import MPI
import numpy as np
import sys

def safe_communication(comm, operation, *args, **kwargs):
    rank = comm.Get_rank()
    print(f"Rank {rank}: Starting safe communication.")
    sys.stdout.flush()
    try:
        result = operation(*args, **kwargs)
        print(f"Rank {rank}: Safe communication succeeded.")
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"Rank {rank}: An error occurred during MPI communication: {e}")
        sys.stdout.flush()
        comm.Abort()



def parallel_odd_even_transposition_sort(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank}: Starting parallel odd-even transposition sort.")
    sys.stdout.flush()

    n = len(data)
    temp_data = np.empty_like(data)

    for i in range(size):
        # Ensure data is contiguous at the start of each iteration
        contiguous_data = np.ascontiguousarray(data)
        print(f"Rank {rank}: Data is contiguous before MPI operation: {contiguous_data.flags['C_CONTIGUOUS']}")
        sys.stdout.flush()

        # Odd phase
        if (i + rank) % 2 == 1:
            if rank < size - 1:
                print(f"Rank {rank}: Sending and receiving in odd phase with rank {rank+1}")
                sys.stdout.flush()
                print(f"Rank {rank}: Sending data with shape {contiguous_data.shape} and dtype {contiguous_data.dtype}")
                sys.stdout.flush()
                comm.Sendrecv(contiguous_data, dest=rank+1, recvbuf=temp_data, source=rank+1)
                # Ensure data is contiguous after MPI operation
                contiguous_temp_data = np.ascontiguousarray(temp_data)
                # Sorting logic here
                for j in range(n-1):
                    if contiguous_temp_data[j] < contiguous_data[j]:
                        contiguous_data[j], contiguous_data[j+1] = contiguous_data[j+1], contiguous_temp_data[j]
                contiguous_data = np.ascontiguousarray(contiguous_data)
                
        # Even phase
        else:
            if rank > 0:
                print(f"Rank {rank}: Sending and receiving in even phase with rank {rank-1}")
                sys.stdout.flush()
                print(f"Rank {rank}: Sending data with shape {contiguous_data.shape} and dtype {contiguous_data.dtype}")
                sys.stdout.flush()
                comm.Sendrecv(contiguous_data, dest=rank-1, recvbuf=temp_data, source=rank-1)
                # Ensure data is contiguous after MPI operation
                contiguous_temp_data = np.ascontiguousarray(temp_data)
                for j in range(1, n):
                    if contiguous_temp_data[j] < contiguous_data[j-1]:
                        contiguous_data[j-1], contiguous_data[j] = contiguous_temp_data[j], contiguous_data[j-1]
                contiguous_data = np.ascontiguousarray(contiguous_data)

        data[:] = contiguous_data
        print(f"Rank {rank}: Data is contiguous after MPI operation: {data.flags['C_CONTIGUOUS']}")
        sys.stdout.flush()

    print(f"Rank {rank}: Finished parallel odd-even transposition sort.")
    sys.stdout.flush()

    return data





def parallel_merge_sort(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_sorted_data = parallel_odd_even_transposition_sort(data, comm)

    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            if rank + step < size:
                recv_data = np.empty_like(data)
                comm.Recv(recv_data, source=rank + step)
                print(f"Rank {rank}: Received data with shape {recv_data.shape} and dtype {recv_data.dtype}")
                sys.stdout.flush()
                recv_data = np.ascontiguousarray(recv_data)
                print(f"Rank {rank}: Data is contiguous after receiving: {recv_data.flags['C_CONTIGUOUS']}")
                sys.stdout.flush()

                local_sorted_data = merge_sorted_arrays(local_sorted_data, recv_data, True)
        elif rank % step == 0:
            local_sorted_data = np.ascontiguousarray(local_sorted_data)
            print(f"Rank {rank}: Data is contiguous before sending: {local_sorted_data.flags['C_CONTIGUOUS']}")
            sys.stdout.flush()

            print(f"Rank {rank}: Sending data with shape {local_sorted_data.shape} and dtype {local_sorted_data.dtype}")
            sys.stdout.flush()
            comm.Send(local_sorted_data, dest=rank - step)
            
            break
        step *= 2

    if rank == 0:
        return local_sorted_data
    else:
        return None




def merge_sorted_arrays(a, b, take_lower):
    """
    Merge two sorted arrays and take the lower or upper half of the merged array,
    depending on the 'take_lower' flag.
    """
    merged = np.empty(len(a) + len(b), dtype=a.dtype)
    i = j = k = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            merged[k] = a[i]
            i += 1
        else:
            merged[k] = b[j]
            j += 1
        k += 1
    while i < len(a):
        merged[k] = a[i]
        i += 1
        k += 1
    while j < len(b):
        merged[k] = b[j]
        j += 1
        k += 1

    # Ensure merged array is contiguous
    merged = np.ascontiguousarray(merged)

    # Decide which half of the array to return
    half = len(merged) // 2
    return merged[:half] if take_lower else merged[half:]



'''def find_splitters(local_data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_sample = np.sort(np.random.choice(local_data, size=10, replace=False))
    local_sample = np.ascontiguousarray(local_sample)

    if rank < size - 1:
        send_max = np.array([local_sample[-1]], dtype=np.float64)
        recv_min = np.array([0.0], dtype=np.float64)
        print(f"Rank {rank}: Sending data with dtype {send_max.dtype}")
        sys.stdout.flush()

        comm.Sendrecv(send_max, dest=rank + 1, recvbuf=recv_min, source=rank + 1)
        local_sample[-1] = recv_min[0]

    if rank > 0:
        send_min = np.array([local_sample[0]], dtype=np.float64)
        recv_max = np.array([0.0], dtype=np.float64)
        print(f"Rank {rank}: Sending data with dtype {send_min.dtype}")
        sys.stdout.flush()
        comm.Sendrecv(send_min, dest=rank - 1, recvbuf=recv_max, source=rank - 1)
        local_sample[0] = recv_max[0]

    all_samples = np.empty(size * 10, dtype=local_sample.dtype)
    comm.Allgather([local_sample, MPI.DOUBLE], [all_samples, MPI.DOUBLE])
    all_samples = np.ascontiguousarray(all_samples)
    print(f"Rank {rank}: Data is contiguous after Allgather: {all_samples.flags['C_CONTIGUOUS']}")
    sys.stdout.flush()

    all_samples.sort()

    potential_splitters = all_samples[::size]

    all_potential_splitters = None
    if rank == 0:
        all_potential_splitters = np.empty(size * len(potential_splitters), dtype=potential_splitters.dtype)
    comm.Gather(potential_splitters, all_potential_splitters, root=0)

    splitters = None
    if rank == 0:
        all_potential_splitters.sort()
        splitters = all_potential_splitters[::size]

    splitters = comm.bcast(splitters, root=0)
    return splitters'''

def find_splitters(local_data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Select a sample from local data
    local_sample = np.sort(np.random.choice(local_data, size=10, replace=False))
    local_sample = np.ascontiguousarray(local_sample)

    # Use parallel sort for the sample
    sorted_sample = parallel_odd_even_transposition_sort(local_sample, comm)
    if rank < size - 1:
        send_max = np.array([local_sample[-1]], dtype=np.float64)
        recv_min = np.array([0.0], dtype=np.float64)
        print(f"Rank {rank}: Sending data with dtype {send_max.dtype}")
        sys.stdout.flush()

        comm.Sendrecv(send_max, dest=rank + 1, recvbuf=recv_min, source=rank + 1)
        local_sample[-1] = recv_min[0]

    if rank > 0:
        send_min = np.array([local_sample[0]], dtype=np.float64)
        recv_max = np.array([0.0], dtype=np.float64)
        print(f"Rank {rank}: Sending data with dtype {send_min.dtype}")
        sys.stdout.flush()
        comm.Sendrecv(send_min, dest=rank - 1, recvbuf=recv_max, source=rank - 1)
        local_sample[0] = recv_max[0]

    all_samples = np.empty(size * 10, dtype=local_sample.dtype)
    comm.Allgather([local_sample, MPI.DOUBLE], [all_samples, MPI.DOUBLE])
    all_samples = np.ascontiguousarray(all_samples)
    print(f"Rank {rank}: Data is contiguous after Allgather: {all_samples.flags['C_CONTIGUOUS']}")
    sys.stdout.flush()

    all_samples.sort()

    potential_splitters = all_samples[::size]

    all_potential_splitters = None
    if rank == 0:
        all_potential_splitters = np.empty(size * len(potential_splitters), dtype=potential_splitters.dtype)
    comm.Gather(potential_splitters, all_potential_splitters, root=0)

    splitters = None
    if rank == 0:
        all_potential_splitters.sort()
        splitters = all_potential_splitters[::size]

    splitters = comm.bcast(splitters, root=0)
    return splitters



def butterfly_alltoallv(send_data_list, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    stages = int(np.log2(size))

    # Initial contiguity check
    print(f"Rank {rank}: Initial data contiguity check: {all(data.flags['C_CONTIGUOUS'] for data in send_data_list)}")
    sys.stdout.flush()

    shapes = [data.shape for data in send_data_list]
    ndims_list = [len(shape) for shape in shapes]
    flattened_data_list = [data.flatten() for data in send_data_list]

    # Ensure flattened_send_data is contiguous
    flattened_send_data = np.concatenate(flattened_data_list) if flattened_data_list else np.array([], dtype=np.float64)
    flattened_send_data = np.ascontiguousarray(flattened_send_data)

    shapes_flattened = np.array([item for shape in shapes for item in shape], dtype=np.int32)
    ndims_flattened = np.array(ndims_list, dtype=np.int32)

    for stage in range(stages):
        bitmask = 1 << stage
        partner = rank ^ bitmask

        if partner < size:
            contiguous_send_data = np.ascontiguousarray(flattened_send_data)
            print(f"Rank {rank}: Data contiguity check before sending: {contiguous_send_data.flags['C_CONTIGUOUS']}")
            sys.stdout.flush()
            
            print(f"Rank {rank}: Sending data with shape {contiguous_send_data.shape} and dtype {contiguous_send_data.dtype}")
            sys.stdout.flush()
            req_send_shapes = comm.Isend(shapes_flattened, dest=partner)
            req_send_ndims = comm.Isend(ndims_flattened, dest=partner)

            status = MPI.Status()
            comm.Probe(source=partner, tag=MPI.ANY_TAG, status=status)
            incoming_shapes_size = status.Get_count(MPI.INT)
            shapes_recv_data = np.empty(incoming_shapes_size, dtype=np.int32)
            req_recv_shapes = comm.Irecv(shapes_recv_data, source=partner)

            comm.Probe(source=partner, tag=MPI.ANY_TAG, status=status)
            incoming_ndims_size = status.Get_count(MPI.INT)
            ndims_recv_data = np.empty(incoming_ndims_size, dtype=np.int32)
            req_recv_ndims = comm.Irecv(ndims_recv_data, source=partner)

            req_send_shapes.Wait()
            req_recv_shapes.Wait()
            req_send_ndims.Wait()
            req_recv_ndims.Wait()

            offsets = np.cumsum([0] + list(ndims_recv_data))
            received_shapes = [tuple(shapes_recv_data[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]

            req_send_data = comm.Isend(contiguous_send_data, dest=partner)
            status = MPI.Status()
            comm.Probe(source=partner, tag=MPI.ANY_TAG, status=status)
            incoming_size = status.Get_count(MPI.DOUBLE)
            print(f"Rank {rank}: Receiving data with expected size {incoming_size}")
            sys.stdout.flush()
            recv_data = np.empty(incoming_size, dtype=flattened_send_data.dtype)
            req_recv_data = comm.Irecv(recv_data, source=partner)

            req_send_data.Wait()
            req_recv_data.Wait()

            contiguous_recv_data = np.ascontiguousarray(recv_data)
            print(f"Rank {rank}: Data contiguity check after receiving: {contiguous_recv_data.flags['C_CONTIGUOUS']}")
            sys.stdout.flush()

            offsets = np.cumsum([0] + [np.prod(shape) for shape in received_shapes])
            reshaped_data_list = [np.reshape(contiguous_recv_data[offsets[i]:offsets[i + 1]], shape) for i, shape in enumerate(received_shapes)]
            print(f"Rank {rank}: Data contiguity check after reshaping: {all(data.flags['C_CONTIGUOUS'] for data in reshaped_data_list)}")
            sys.stdout.flush()

            # Ensure reshaped data list is contiguous before subsequent use
            reshaped_data_list = [np.ascontiguousarray(data) for data in reshaped_data_list]
            
            send_data_list = reshaped_data_list

    # Final contiguity check
    print(f"Rank {rank}: Final data contiguity check: {all(data.flags['C_CONTIGUOUS'] for data in send_data_list)}")
    sys.stdout.flush()

    return send_data_list


#Multidimensional Arrays

def sample_sort(local_data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not isinstance(local_data, np.ndarray):
        raise ValueError("local_data must be a numpy array")

    # Initial sorting
    local_sorted_data = parallel_odd_even_transposition_sort(local_data, comm)

    # After initial sort
    print(f"Rank {rank}: Data contiguity check after initial sort: {local_sorted_data.flags['C_CONTIGUOUS']}")
    sys.stdout.flush()

    splitters = find_splitters(local_sorted_data, comm)

    # After find_splitters
    print(f"Rank {rank}: Data contiguity check after find_splitters: {local_sorted_data.flags['C_CONTIGUOUS']}")
    sys.stdout.flush()

    indices = np.searchsorted(local_sorted_data, splitters)
    partitioned_data = np.split(local_sorted_data, indices)

    # Ensure each part of partitioned_data is contiguous
    partitioned_data = [np.ascontiguousarray(d) for d in partitioned_data]

    # Before butterfly_alltoallv
    print(f"Rank {rank}: Data contiguity check before butterfly_alltoallv: {all(d.flags['C_CONTIGUOUS'] for d in partitioned_data)}")
    sys.stdout.flush()

    exchanged_data_list = butterfly_alltoallv(partitioned_data, comm)

    # After butterfly_alltoallv
    print(f"Rank {rank}: Data contiguity check after butterfly_alltoallv: {all(d.flags['C_CONTIGUOUS'] for d in exchanged_data_list)}")
    sys.stdout.flush()

    local_sorted_data = np.concatenate(exchanged_data_list) if exchanged_data_list else np.array([], dtype=local_sorted_data.dtype)

    # Ensure local_sorted_data is contiguous before final sort
    local_sorted_data = np.ascontiguousarray(local_sorted_data)

    # Final sort
    local_sorted_data = parallel_odd_even_transposition_sort(local_sorted_data, comm)

    # After the final sort
    print(f"Rank {rank}: Data contiguity check at the end of sample_sort: {local_sorted_data.flags['C_CONTIGUOUS']}")
    sys.stdout.flush()

    return local_sorted_data


  

'''if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test with different data sizes and distributions
    for test_size in [100, 1000, 10000]:
        for distribution in ['uniform', 'normal', 'constant']:
            # Initialize local data based on the distribution
            if distribution == 'uniform':
                local_data = np.random.randint(0, 100000, size=test_size)
            elif distribution == 'normal':
                local_data = np.random.normal(loc=50000, scale=10000, size=test_size).astype(np.int32)
            else:
                local_data = np.full(test_size, 50000, dtype=np.int32)

            # Print the type of test being conducted
            if rank == 0:
                print(f"\nTesting with size {test_size} and {distribution} distribution")
                print("Generated input list:", local_data)
            sys.stdout.flush()

            # Start timing
            start_time = MPI.Wtime()

            try:
                # Perform sorting
                local_sorted_data = sample_sort(local_data, comm)

                # End timing
                end_time = MPI.Wtime()

                # Print the runtime
                if rank == 0:
                    print(f"Sorting completed in {end_time - start_time} seconds")
                sys.stdout.flush()

                # Print the sorted list
                print(f"Rank {rank} sorted data:", local_sorted_data)
                sys.stdout.flush()

            except Exception as e:
                # Print any exceptions that occur
                print(f"Rank {rank}: An error occurred during the sorting process: {e}")
                sys.stdout.flush()
                comm.Abort()'''

'''if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set seed for reproducibility
    np.random.seed(0)

    performance_metrics = {}  # Dictionary to store performance metrics

    # Test with different data sizes and distributions
    for test_size in [100, 1000, 10000]:
        for distribution in ['uniform', 'normal', 'constant']:
            # Initialize local data based on the distribution
            if distribution == 'uniform':
                local_data = np.random.randint(0, 100000, size=test_size)
            elif distribution == 'normal':
                local_data = np.random.normal(loc=50000, scale=10000, size=test_size).astype(np.int32)
            else:
                local_data = np.full(test_size, 50000, dtype=np.int32)

            # Print the type of test being conducted
            if rank == 0:
                print(f"\nTesting with size {test_size} and {distribution} distribution")

            # Start timing
            start_time = MPI.Wtime()

            try:
                # Perform sorting
                local_sorted_data = sample_sort(local_data, comm)

                # End timing
                end_time = MPI.Wtime()

                # Print the runtime
                if rank == 0:
                    print(f"Sorting completed in {end_time - start_time} seconds with {size} processes")

            except Exception as e:
                # Print any exceptions that occur
                print(f"Rank {rank}: An error occurred during the sorting process: {e}")
                comm.Abort()'''

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(0)
    performance_metrics = {}

    for test_size in [100, 1000, 10000]:
        for distribution in ['uniform', 'normal', 'constant']:
            local_data = np.random.randint(0, 100000, size=test_size) if distribution == 'uniform' else \
                         np.random.normal(loc=50000, scale=10000, size=test_size).astype(np.int32) if distribution == 'normal' else \
                         np.full(test_size, 50000, dtype=np.int32)

            if rank == 0:
                print(f"\nTesting with size {test_size}, distribution {distribution}, and {size} processes")

            comm.Barrier()  # Synchronize before starting timing
            start_time = MPI.Wtime()

            try:
                local_sorted_data = sample_sort(local_data, comm)
            except Exception as e:
                print(f"Rank {rank}: An error occurred during the sorting process: {e}")
                comm.Abort()

            end_time = MPI.Wtime()
            runtime = end_time - start_time

            if rank == 0:
                print(f"Sorting completed in {runtime} seconds with {size} processes")
                performance_metrics[(test_size, distribution, size)] = runtime

    if rank == 0:
        for key, runtime in performance_metrics.items():
            print(f"Size: {key[0]}, Distribution: {key[1]}, Processes: {key[2]}, Runtime: {runtime} seconds")
