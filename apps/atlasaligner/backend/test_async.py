import asyncio

async def fetch_data():
    print("Start fetching data...")
    await asyncio.sleep(2)  # Simulates waiting for an I/O operation
    print("Data fetched!")
    return "data"

async def process_data():
    print("Start processing...")
    await asyncio.sleep(1)  # Simulates another I/O operation
    print("Processing done!")

async def main():
    # Run both tasks concurrently
    task1 = asyncio.create_task(fetch_data())
    task2 = asyncio.create_task(process_data())
    print('returned')
    # Wait for both tasks to complete
    await task1
    await task2

asyncio.run(main())
