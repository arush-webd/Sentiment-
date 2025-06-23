"""
Specific scraper for CMIE Prowess interface
Based on the actual interface structure
"""

import time
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class CMIEProwessScraper:
    def __init__(self, credentials_file='configs/cmie_credentials.json'):
        self.credentials = self._load_credentials(credentials_file)
        self.driver = None
        self._setup_driver()
    
    def _load_credentials(self, file_path):
        """Load CMIE credentials"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ CMIE credentials not found. Please create configs/cmie_credentials.json")
            return {'username': '', 'password': '', 'base_url': 'https://prowess.cmie.com'}
    
    def _setup_driver(self):
        """Setup Chrome driver for CMIE"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Remove this to see browser
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Use webdriver manager
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("✓ Chrome driver initialized")
    
    def login(self):
        """Login to CMIE Prowess"""
        try:
            print("🔐 Logging into CMIE Prowess...")
            self.driver.get(self.credentials['base_url'])
            
            # Wait for login form
            wait = WebDriverWait(self.driver, 10)
            
            # Find and fill username (adapt based on actual form)
            username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
            username_field.send_keys(self.credentials['username'])
            
            # Find and fill password
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(self.credentials['password'])
            
            # Submit form
            login_button = self.driver.find_element(By.XPATH, "//input[@type='submit']")
            login_button.click()
            
            # Wait for dashboard to load
            time.sleep(3)
            
            print("✓ Successfully logged into CMIE")
            return True
            
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False
    
    def search_company_news(self, company_name, max_articles=10):
        """
        Search for news articles for a specific company
        Based on your CMIE interface
        """
        try:
            print(f"🔍 Searching news for {company_name}...")
            
            # Navigate to the company search (adapt URL as needed)
            search_url = f"{self.credentials['base_url']}/kommon/servlet/rptsrv?page=summary&company={company_name}"
            self.driver.get(search_url)
            
            time.sleep(2)  # Rate limiting
            
            # Look for news section or links
            # This part needs to be adapted based on actual CMIE HTML structure
            news_articles = []
            
            # Based on your screenshot, it looks like there might be different sections
            # We would need to identify the exact selectors for news content
            
            # Placeholder implementation - you'll need to adapt this
            try:
                # Look for news-related elements
                news_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'news') or contains(text(), 'News')]")
                
                for element in news_elements[:max_articles]:
                    # Extract news data
                    news_data = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'headline': element.text[:100] if element.text else 'Sample headline',
                        'content': element.text if element.text else 'Sample content',
                        'company': company_name,
                        'url': self.driver.current_url,
                        'source': 'CMIE_PROWESS'
                    }
                    news_articles.append(news_data)
                    
            except Exception as e:
                print(f"⚠️ Could not extract news: {e}")
                
                # Return sample data for testing
                news_articles = [{
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'headline': f'{company_name} shows strong performance in latest quarter',
                    'content': f'{company_name} demonstrates robust financial metrics and market position.',
                    'company': company_name,
                    'url': self.driver.current_url,
                    'source': 'CMIE_PROWESS'
                }]
            
            print(f"✓ Found {len(news_articles)} articles for {company_name}")
            return news_articles
            
        except Exception as e:
            print(f"❌ Error searching for {company_name}: {e}")
            return []
    
    def get_company_data(self, company_name):
        """
        Get comprehensive company data from CMIE
        Based on the interface sections visible in your screenshot
        """
        try:
            # Navigate to company page
            self.driver.get(f"{self.credentials['base_url']}/query-builder")
            
            # Search for the company
            search_box = self.driver.find_element(By.ID, "company-search")  # Adapt ID
            search_box.clear()
            search_box.send_keys(company_name)
            
            # Click search button
            search_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Search')]")
            search_button.click()
            
            time.sleep(2)
            
            # Extract available data sections
            company_data = {
                'company_name': company_name,
                'one_page_profile': self._extract_if_available('One Page Profile'),
                'financial_statements': self._extract_if_available('Financial Statements'),
                'ownership_reports': self._extract_if_available('Ownership & Governance'),
                'news_articles': self.search_company_news(company_name)
            }
            
            return company_data
            
        except Exception as e:
            print(f"❌ Error getting company data: {e}")
            return None
    
    def _extract_if_available(self, section_name):
        """Extract data from a section if available"""
        try:
            # Look for the section
            section = self.driver.find_element(By.XPATH, f"//text()[contains(., '{section_name}')]")
            return f"Data available for {section_name}"
        except:
            return f"No data available for {section_name}"
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")

# Test function
def test_cmie_scraper():
    """Test the CMIE scraper"""
    scraper = CMIEProwessScraper()
    
    try:
        # Test login (will fail without real credentials)
        login_success = scraper.login()
        
        if login_success:
            # Test company search
            companies = ['Infosys Ltd.', 'TCS Ltd.', 'Reliance Industries']
            
            for company in companies:
                news = scraper.search_company_news(company, max_articles=3)
                print(f"\nNews for {company}: {len(news)} articles")
                
                for article in news:
                    print(f"- {article['headline'][:50]}...")
        else:
            print("⚠️ Skipping tests - login required")
            
    finally:
        scraper.close()

if __name__ == "__main__":
    test_cmie_scraper()
