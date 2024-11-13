import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        async def delay_route(route, request):
            await asyncio.sleep(5)  # Delay each request by 5 seconds
            await route.continue_()

        await page.route("**/*", delay_route)

        await page.goto("https://www.thecrag.com/photo/10576834203")
        await page.screenshot(path="screenshot.png")
        await browser.close()

asyncio.run(run())