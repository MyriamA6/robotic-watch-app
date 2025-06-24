import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import streamlit as st
import time
from itertools import cycle

def website_retrieval():
    # webscraping to retrieve the news solely from the website "https://www.humanoidsdaily.com/"
    init_url = "https://www.humanoidsdaily.com/"
    response = requests.get(init_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    news=dict()

    body = soup.find("body")

    section = body.find("section")

    main_section = section.find("main")

    main_div = main_section.find("div", class_="divide-y divide-gray-200 dark:divide-gray-700")

    main_block = main_div.find("ul")

    news_listed = main_block.find_all("li")

    cpt = 1
    for li in news_listed:

        # Retrieving the whole article
        article = li.find("article")

        # We start by retrieving the image of the article
        div_image = article.find("div", class_="mb-6 xl:hidden")

        image_tag = div_image.find("img")

        image = urljoin(init_url, image_tag['src']) if image_tag and image_tag.has_attr('src') else None

        # We find its date
        div_date = article.find("dl")

        dd_date = div_date.find("dd")

        time_date = div_date.find("time")

        date = time_date.text

        # Then we look for its title and url
        div_info = article.find("div", class_="mr-4 space-y-5 xl:col-span-2")

        div_title = div_info.find("div", class_="space-y-2")

        sub_div_title = div_title.find("div")

        sub_div_title_h2 = sub_div_title.find("h2")

        sub_div_title_h2_a = sub_div_title_h2.find("a")

        title = sub_div_title_h2_a.text

        link = urljoin(init_url, sub_div_title_h2_a.attrs["href"])

        news[cpt] = {
            "title": title,
            "link": link,
            "date": date,
            "image": image
        }

        cpt += 1


    return news


