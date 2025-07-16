from rating.rate_manager import NineFactorRater

if __name__ == '__main__':
    rater = NineFactorRater(host='10.26.0.8', port=2000, username='hmcz', password='Hmcz_12345678')
    rater.rate_all(1)