import pickle
import os
import struct
from torch_geometric.loader import DataLoader

class BackgroundLoader:
    def __init__(self, dataset, confidence_dataset, batch_size):
        self.batch_size = batch_size
        self.has_confidence = confidence_dataset is not None

        read_pipe, write_pipe = os.pipe()
        read_pipe = os.fdopen(read_pipe, 'rb')
        write_pipe = os.fdopen(write_pipe, 'wb')

        pid = os.fork()
        if pid:
            write_pipe.close()
            self.read_pipe = read_pipe
        else:
            read_pipe.close()
            background_task(dataset, confidence_dataset, write_pipe)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        next_batch = []
        next_batch_confidence = []
        for i in range(0, self.batch_size):
            print('Processing ' + str(self.idx))
            self.idx += 1

            item = get_next_item(self.read_pipe)
            if item:
                if self.has_confidence:
                    next_batch.append(item[0])
                    next_batch_confidence.append(item[1])
                else:
                    next_batch.append(item)
            else:
                break

        if not next_batch:
            raise StopIteration
    
        if self.has_confidence:
            return next(iter(DataLoader(next_batch, batch_size=len(next_batch), shuffle=False))), next(iter(DataLoader(next_batch_confidence, batch_size=len(next_batch_confidence), shuffle=False)))
        else:
            return next(iter(DataLoader(next_batch, batch_size=len(next_batch), shuffle=False)))

def get_next_item(pipe):
    num_bytes = struct.unpack('=Q', pipe.read(8))[0]
    if num_bytes == 0:
        return None
    else:
        return pickle.loads(pipe.read(num_bytes))

def background_task(dataset, confidence_dataset, pipe):
    for idx, graph in enumerate(dataset):
        print('Background task: preprocessing ' + str(idx))
        if not graph:
            print('Skipping ' + str(idx) + ' due to preprocessing fail')
            continue

        if confidence_dataset:
            confidence_graph = confidence_dataset.get_by_name(graph.name)
            if not confidence_graph:
                print('Skipping ' + str(idx) + ' due to confidence preprocessing fail')
                continue
            out_data = (graph, confidence_graph)
        else:
            out_data = graph

        out_data = pickle.dumps(out_data)
        pipe.write(struct.pack('=Q', len(out_data)))
        pipe.write(out_data)
        pipe.flush()

    print('Background task complete!')
    pipe.write(struct.pack('=Q', 0))
    pipe.flush()
    os._exit(0)
