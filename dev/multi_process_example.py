from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
from multiprocessing import Process, Queue
from typing import List


from pipeline.audio_diarization import simulated_diarization
from pipeline.audio_detect import AudioRecognizer


def my_function(x: int) -> int:
    print(f"Function running task(x): {x}")
    return x * x


with ThreadPoolExecutor(max_workers=3) as executors:
    futures = [executors.submit(my_function, i) for i in range(9)]

    for future in as_completed(futures):
        result = future.result()
        print(f"Result: {result}")


def heavy_task(n):
    return sum(i * i for i in range(n))


with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(heavy_task, [10_000_000, 20_000_000, 30_000_000]))
    print(results)

# with ProcessPoolExecutor(max_workers=5) as executor:
#     futures = []
#     for _ in range(5):
#         futures.append(executor.submit(simulated_diarization,"./2 people conversation.opus"))
#
#     for future in futures:
#


def simulated_diarization_mp(path_to_audio, queue: Queue):
    for segment, person in simulated_diarization(path_to_audio):
        queue.put((segment, person))
    queue.put((None, None))


def simulate():
    # queue = Queue()
    processes: List[Process] = []
    num_processes = 2

    for _ in range(num_processes):
        ar = AudioRecognizer()
        process = Process(
            # target=simulated_diarization_mp,
            target=ar.whisper_model,
            # args=("./2 people conversation.opus", queue),
            args=("./2 people conversation.opus",),
        )
        process.start()
        processes.append(process)

    # finished = 0
    # while finished < num_processes:
    #     segment, person = queue.get()
    #     if segment is None and person is None:
    #         finished += 1
    #         continue
    #     yield segment, person

    for process in processes:
        process.join()


# Example usage:
# for data in simulate():
#     print(data)
simulate()


def blocking_task(x):
    import time

    time.sleep(1)
    return x * x


async def main():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, blocking_task, 5)
        print(result)


asyncio.run(main())
